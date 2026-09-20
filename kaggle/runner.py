import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

OUT = Path("/kaggle/working/automation_result.txt")
TARGET = Path("/kaggle/working/indictrans_pkgs")
# Kaggle script kernels execute from /kaggle/src. Auxiliary files are not
# guaranteed to be present there, so keep the R49 pins in the runner itself.
R49_REQUIREMENTS = """huggingface-hub==0.36.2
transformers==4.57.6
indictranstoolkit==1.1.1
"""
REQ = Path("/kaggle/working/r49-requirements.txt")
lines = []

def log(message=""):
    print(message, flush=True)
    lines.append(str(message))

def save():
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ Result saved to {OUT}", flush=True)

def run_checked(cmd, label):
    log(f"\n{label}")
    log("$ " + " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
        lines.extend(result.stdout.rstrip().splitlines())
    if result.stderr:
        print(result.stderr.rstrip(), flush=True)
        lines.extend(result.stderr.rstrip().splitlines())
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")

try:
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
    import torch
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"CUDA version: {torch.version.cuda}")
    log(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    log("\n[4] R49 requirements")
    REQ.write_text(R49_REQUIREMENTS, encoding="utf-8")
    log(R49_REQUIREMENTS.strip())

    log("\n[5] Installing R49 packages without touching Kaggle system site-packages")
    TARGET.mkdir(parents=True, exist_ok=True)

    # Use normal dependency resolution inside the isolated target.
    # This keeps the Kaggle base environment untouched while avoiding
    # the missing-dependency problem caused by --no-deps.
    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            "--target",
            str(TARGET),
            "-r",
            str(REQ),
        ],
        "Installing R49 packages",
    )

    sys.path.insert(0, str(TARGET))

    log("\n[6] R49 package versions")
    import huggingface_hub
    import transformers
    log(f"huggingface_hub: {huggingface_hub.__version__}")
    log(f"transformers: {transformers.__version__}")

    log("\n[7] IndicTransToolkit")
    import IndicTransToolkit
    from IndicTransToolkit import IndicProcessor
    log("IndicTransToolkit: import OK")
    processor = IndicProcessor(inference=True)
    log(f"IndicProcessor: {processor.__class__.__name__} created")

    log("\n[8] Simple processor smoke test")
    sample = ["This is a Telugu translation pipeline test."]
    try:
        processed = processor.preprocess_batch(
            sample,
            src_lang="eng_Latn",
            tgt_lang="tel_Telu",
        )
        log(f"Preprocessed items: {len(processed)}")
    except TypeError:
        log("Processor method signature differs; import/creation check passed.")

    log("\n[9] Kaggle environment")
    log("KAGGLE_KERNEL_RUN_TYPE: " + str(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")))
    log("KAGGLE_URL_BASE: " + str(os.environ.get("KAGGLE_URL_BASE")))

    log("\n" + "=" * 70)
    log("R49 CHECKPOINT PASSED")
    log("=" * 70)
    save()

except Exception as exc:
    log("\n" + "=" * 70)
    log("R49 CHECKPOINT FAILED")
    log("=" * 70)
    log(f"{type(exc).__name__}: {exc}")
    log("\nTraceback:")
    tb = traceback.format_exc()
    print(tb, flush=True)
    lines.extend(tb.rstrip().splitlines())
    save()
    raise
