import os
import platform
import subprocess

print("=" * 70)
print("TELUGU DUBBING AI — GITHUB → KAGGLE GPU TEST")
print("=" * 70)

print("\n[1] Python")
print(platform.python_version())

print("\n[2] Platform")
print(platform.platform())

print("\n[3] GPU")

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
    print(result.stdout.strip())
else:
    print("❌ NVIDIA GPU not detected")

print("\n[4] Environment")
print("KAGGLE_KERNEL_RUN_TYPE:", os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
print("KAGGLE_URL_BASE:", os.environ.get("KAGGLE_URL_BASE"))

print("\n" + "=" * 70)
print("HELLO FROM GITHUB → KAGGLE GPU")
print("=" * 70)
