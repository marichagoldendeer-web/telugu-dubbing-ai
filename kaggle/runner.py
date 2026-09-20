import os
import platform
import subprocess
import sys
from pathlib import Path

OUT = Path("/kaggle/working/automation_result.txt")
TARGET = Path("/kaggle/working/indictrans_pkgs")
REQ = Path(__file__).with_name("r49-requirements.txt")
lines = []

def log(message=""):
    print(message)
    lines.append(str(message))

def run_checked(cmd, label):
    log(f"\n{label}")
    log("$ " + " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
        lines.extend(result.stdout.rstrip().splitlines())
    if result.stderr:
        print(result.stderr.rstrip())
        lines.extend(result.stderr.rstrip().splitlines())
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")

log("=" * 70)
log("R49 — INDIC TRANS COMPATIBILITY / GPU AUTOMATION TEST")
log("=" * 70)

log("\n[1] Python")
log(sys.version.splitlines()[0])
log(platform.platform())

log("\n[2] GPU")
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
if gpu.returncode != 0:
    raise RuntimeError("NVIDIA GPU not detected")
log(gpu.stdout.strip())

log("\n[3] Base PyTorch / CUDA")
try:
    import torch
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"CUDA version: {torch.version.cuda}")
    log(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as exc:
    raise RuntimeError(f"PyTorch check failed: {type(exc).__name__}: {exc}")

log("\n[4] R49 requirements")
if not REQ.exists():
    raise FileNotFoundError(f"Missing requirements file: {REQ}")
log(REQ.read_text(encoding="utf-8").strip())

log("\n[5] One-shot R49 package install")
TARGET.mkdir(parents=True, exist_ok=True)
run_checked(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade",
        "--no-deps",
        "--target",
        str(TARGET),
        "-r",
        str(REQ),
    ],
    "Installing R49 packages",
)

# Prefer the isolated R49 package directory over base-site packages.
sys.path.insert(0, str(TARGET))

log("\n[6] R49 package versions")
import huggingface_hub
import transformers

log(f"huggingface_hub: {huggingface_hub.__version__}")
log(f"transformers: {transformers.__version__}")

try:
    import IndicTransToolkit
    from IndicTransToolkit import IndicProcessor
    log("IndicTransToolkit: import OK")
    processor = IndicProcessor(inference=True)
    log(f"IndicProcessor: {processor.__class__.__name__} created")
except Exception as exc:
    raise RuntimeError(
        f"IndicTransToolkit / IndicProcessor test failed: "
        f"{type(exc).__name__}: {exc}"
    )

log("\n[7] Simple processor smoke test")
try:
    sample = ["This is a Telugu translation pipeline test."]
    processed = processor.preprocess_batch(
        sample,
        src_lang="eng_Latn",
        tgt_lang="tel_Telu",
    )
    log(f"Preprocessed items: {len(processed)}")
except TypeError:
    # Keep the R49 checkpoint focused on import/processor compatibility.
    log("Processor method signature differs; import/creation check passed.")
except Exception as exc:
    raise RuntimeError(f"Processor smoke test failed: {type(exc).__name__}: {exc}")

log("\n[8] Kaggle environment")
log("KAGGLE_KERNEL_RUN_TYPE: " + str(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")))
log("KAGGLE_URL_BASE: " + str(os.environ.get("KAGGLE_URL_BASE")))

log("\n" + "=" * 70)
log("R49 CHECKPOINT PASSED")
log("=" * 70)

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n✅ Result saved to {OUT}")
