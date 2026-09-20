import asyncio
import subprocess
import sys
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
LOG = WORK / "r64_diagnostic.txt"
OUTPUT = WORK / "r64_english_tts.mp3"

TEST_TEXT = (
    "This is the R64 English text to speech diagnostic. "
    "The Telugu speech recognition and translation pipeline passed R63. "
    "This step verifies that the translated English text can be converted "
    "into a playable English voice track."
)
VOICE = "en-US-GuyNeural"

def log(msg=""):
    msg = str(msg)
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def run_cmd(args, timeout=300):
    log("$ " + " ".join(map(str, args)))
    try:
        p = subprocess.run(
            args, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout, check=False
        )
        if p.stdout:
            for line in p.stdout.rstrip().splitlines():
                log(line)
        log(f"[returncode={p.returncode}]")
        return p.returncode
    except Exception as exc:
        log(f"COMMAND_EXCEPTION {type(exc).__name__}: {exc}")
        return 999

async def synthesize():
    import edge_tts
    communicate = edge_tts.Communicate(TEST_TEXT, VOICE)
    await communicate.save(str(OUTPUT))

try:
    LOG.write_text("", encoding="utf-8")
    log("=" * 78)
    log("R64 — ENGLISH TEXT TO SPEECH DIAGNOSTIC")
    log("=" * 78)

    log("\n[1] RUNTIME")
    log(f"PYTHON={sys.version}")
    log(f"EXECUTABLE={sys.executable}")
    run_cmd([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader"
    ])

    log("\n[2] INSTALL PINNED TTS RUNTIME")
    rc = run_cmd([
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check", "--no-cache-dir",
        "edge-tts==7.2.7"
    ], timeout=300)
    if rc != 0:
        raise RuntimeError("edge-tts installation failed")

    import edge_tts
    log(f"EDGE_TTS={getattr(edge_tts, '__version__', 'unknown')}")
    log(f"VOICE={VOICE}")
    log(f"TEXT={TEST_TEXT}")

    log("\n[3] SYNTHESIZE ENGLISH SPEECH")
    asyncio.run(synthesize())

    if not OUTPUT.exists():
        raise RuntimeError("TTS output file was not created")
    if OUTPUT.stat().st_size < 1000:
        raise RuntimeError("TTS output file is unexpectedly small")

    log(f"TTS_OUTPUT={OUTPUT}")
    log(f"TTS_OUTPUT_BYTES={OUTPUT.stat().st_size}")

    log("\n[4] VERIFY AUDIO")
    rc = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,format_name",
        "-of", "default=noprint_wrappers=1",
        str(OUTPUT)
    ], timeout=120)
    if rc != 0:
        raise RuntimeError("ffprobe could not validate TTS output")

    log("\nR64 CHECKPOINT 1 PASSED — ENGLISH TTS")
    log("R64 DIAGNOSTIC COMPLETE — ENGLISH TTS PASSED")

except Exception as exc:
    log("\n" + "=" * 78)
    log("R64 DIAGNOSTIC FAILED")
    log("=" * 78)
    log(f"{type(exc).__name__}: {exc}")
    log(traceback.format_exc())
    raise
