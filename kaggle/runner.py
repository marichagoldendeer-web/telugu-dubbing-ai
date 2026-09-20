import os
import platform
import subprocess
from pathlib import Path

OUT = Path("/kaggle/working/automation_result.txt")
lines = []

def log(message=""):
    print(message)
    lines.append(str(message))

log("=" * 70)
log("TELUGU DUBBING AI — GITHUB → KAGGLE GPU TEST")
log("=" * 70)

log("\n[1] Python")
log(platform.python_version())

log("\n[2] Platform")
log(platform.platform())

log("\n[3] GPU")
result = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader"
    ],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    log(result.stdout.strip())
else:
    log("❌ NVIDIA GPU not detected")
    log(result.stderr.strip())

log("\n[4] PyTorch / CUDA")
try:
    import torch
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA version: {torch.version.cuda}")
        log(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as exc:
    log(f"PyTorch check unavailable: {type(exc).__name__}: {exc}")

log("\n[5] Environment")
log("KAGGLE_KERNEL_RUN_TYPE: " + str(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")))
log("KAGGLE_URL_BASE: " + str(os.environ.get("KAGGLE_URL_BASE")))

log("\n" + "=" * 70)
log("HELLO FROM GITHUB → KAGGLE GPU")
log("=" * 70)

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n✅ Result saved to {OUT}")
