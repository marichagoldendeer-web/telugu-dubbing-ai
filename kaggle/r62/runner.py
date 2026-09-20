import platform
import subprocess
import sys
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
LOG = WORK / "r62_diagnostic.txt"
VIDEO_DS = "anjiallkumar/r52-telugu-source-video"
ASR_DS = "anjiallkumar/indicconformer-te-hybrid-rnnt-large"

def log(msg=""):
    msg = str(msg)
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def run_cmd(args, cwd=None, timeout=None):
    log("$ " + " ".join(map(str, args)))
    try:
        p = subprocess.run(
            args, cwd=cwd, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=False
        )
        if p.stdout:
            for line in p.stdout.rstrip().splitlines():
                log(line)
        log(f"[returncode={p.returncode}]")
        return p.returncode
    except Exception as exc:
        log(f"COMMAND_EXCEPTION {type(exc).__name__}: {exc}")
        return 999

def locate(slug, marker=None):
    name = slug.split("/", 1)[1]
    base = Path("/kaggle/input")
    for p in [base / name, base / slug]:
        if p.exists():
            return p
    if not base.exists():
        return None
    direct = [p for p in base.rglob("*") if p.is_dir() and p.name == name]
    if direct:
        return direct[0]
    if marker:
        hits = [p.parent for p in base.rglob(marker) if p.is_file()]
        if hits:
            return hits[0]
    return None

try:
    LOG.write_text("", encoding="utf-8")
    log("=" * 78)
    log("R62 — KAGGLE LOG STREAM + CONTROLLED NEMO/INDICCONFORMER DIAGNOSTIC")
    log("=" * 78)

    log("\n[1] RUNTIME")
    log(f"PYTHON={sys.version}")
    log(f"EXECUTABLE={sys.executable}")
    log(f"PLATFORM={platform.platform()}")
    run_cmd([sys.executable, "-m", "pip", "--version"])
    run_cmd([sys.executable, "-m", "pip", "show", "torch", "numpy", "setuptools", "transformers", "huggingface_hub", "pytorch-lightning"])

    log("\n[2] GPU")
    run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])

    log("\n[3] DATASETS")
    video_root = locate(VIDEO_DS, "*.mp4")
    asr_root = locate(ASR_DS, "*.nemo")
    log(f"VIDEO_ROOT={video_root}")
    log(f"ASR_ROOT={asr_root}")
    if not video_root or not asr_root:
        raise FileNotFoundError("Required R53-verified dataset missing")
    videos = sorted(video_root.rglob("*.mp4"))
    nemos = sorted(asr_root.rglob("*.nemo"))
    if not videos or not nemos:
        raise FileNotFoundError("MP4 or .nemo file missing")
    video, nemo_model_path = videos[0], nemos[0]
    log(f"VIDEO={video}")
    log(f"NEMO_MODEL={nemo_model_path}")

    log("\n[4] CLONE AI4BHARAT NEMO")
    nemo_repo = WORK / "NeMo"
    if nemo_repo.exists():
        run_cmd(["git", "rev-parse", "HEAD"], cwd=str(nemo_repo))
    else:
        rc = run_cmd([
            "git", "clone", "--depth", "1", "--branch", "nemo-v2",
            "https://github.com/AI4Bharat/NeMo.git", str(nemo_repo)
        ], timeout=300)
        if rc != 0:
            raise RuntimeError("NeMo clone failed")

    log("\n[5] NE MO METADATA")
    run_cmd(["git", "rev-parse", "HEAD"], cwd=str(nemo_repo))
    run_cmd([sys.executable, "-m", "pip", "check"])

    log("\n[6] CONTROLLED PIP RESOLUTION DRY-RUN")
    log("This step does NOT modify the environment.")
    dry_rc = run_cmd([
        sys.executable, "-m", "pip", "install",
        "-e", ".[asr]",
        "--dry-run",
        "--report", str(WORK / "r62_pip_report.json")
    ], cwd=str(nemo_repo), timeout=900)
    if dry_rc != 0:
        log("R62 CHECKPOINT — NEMO DEPENDENCY RESOLUTION FAILED")
        raise RuntimeError("pip dry-run failed; see r62_diagnostic.txt")

    log("NEMO_DEPENDENCY_DRY_RUN_PASSED")

    log("\n[7] INSTALL KNOWN COMPATIBILITY BASELINE")
    rc = run_cmd([
        sys.executable, "-m", "pip", "install",
        "Cython", "packaging", "numpy==1.26.4", "pyarrow<21.0.0"
    ], timeout=600)
    if rc != 0:
        raise RuntimeError("baseline dependency install failed")

    log("\n[8] INSTALL AI4BHARAT NEMO ASR")
    rc = run_cmd([
        sys.executable, "-m", "pip", "install", "-e", ".[asr]"
    ], cwd=str(nemo_repo), timeout=1200)
    if rc != 0:
        raise RuntimeError("NeMo ASR installation failed")

    log("NEMO_INSTALL_PASSED")

    log("\n[8B] RE-APPLY NUMPY/PYARROW COMPATIBILITY")
    rc = run_cmd([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall", "numpy==1.26.4", "pyarrow<21.0.0"
    ], timeout=600)
    if rc != 0:
        raise RuntimeError("post-NeMo numpy/pyarrow compatibility install failed")

    run_cmd([sys.executable, "-m", "pip", "check"])

    log("\n[9] IMPORT TEST")
    log(f"NE MO_SOURCE={nemo_repo}")
    log(f"PYTHONPATH_BEFORE={sys.path}")
    if str(nemo_repo) not in sys.path:
        sys.path.insert(0, str(nemo_repo))
    log(f"PYTHONPATH_AFTER={sys.path}")
    import importlib.util
    log(f"NEMO_SPEC={importlib.util.find_spec('nemo')}")

    import torch
    log(f"TORCH={torch.__version__}")
    log(f"TORCH_CUDA={torch.version.cuda}")
    log(f"CUDA_AVAILABLE={torch.cuda.is_available()}")

    # AI4Bharat NeMo's exp_manager imports NeptuneLogger unconditionally.
    # Newer PyTorch Lightning releases removed that logger, but NeMo does
    # not need it for this inference path. Provide a minimal compatibility
    # shim so the ASR collection can import without enabling Neptune.
    import pytorch_lightning
    log(f"PYTORCH_LIGHTNING={pytorch_lightning.__version__}")
    import pytorch_lightning.loggers as pl_loggers
    if not hasattr(pl_loggers, "NeptuneLogger"):
        class NeptuneLoggerCompat:
            pass
        pl_loggers.NeptuneLogger = NeptuneLoggerCompat
        log("NEPTUNELOGGER_SHIM=APPLIED")
    else:
        log("NEPTUNELOGGER_SHIM=NOT_NEEDED")

    import nemo
    log(f"NEMO_FILE={getattr(nemo, '__file__', None)}")
    import nemo.collections.asr as nemo_asr
    log(f"NEMO_ASR_IMPORT_PASSED={nemo_asr.__name__}")

    log("\n[10] MODEL RESTORE TEST")
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo_model_path), map_location="cuda"
    )
    log(f"MODEL_CLASS={type(model).__name__}")
    log("MODEL_RESTORE_PASSED")

    log("\n[11] 30-SECOND AUDIO + STT SMOKE")
    smoke = WORK / "r62_smoke_30s.wav"
    rc = run_cmd([
        "ffmpeg", "-y", "-ss", "0", "-t", "30", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(smoke)
    ], timeout=180)
    if rc != 0 or not smoke.exists():
        raise RuntimeError("ffmpeg smoke extraction failed")

    model.freeze()
    model = model.to("cuda")
    errors = []
    text_out = ""
    for decoder in ["ctc", "rnnt"]:
        try:
            if hasattr(model, "cur_decoder"):
                model.cur_decoder = decoder
            log(f"TRY_DECODER={decoder}")
            try:
                result = model.transcribe(
                    [str(smoke)], batch_size=1, logprobs=False, language_id="te"
                )
            except TypeError:
                result = model.transcribe([str(smoke)], batch_size=1)
            item = result[0]
            text_out = str(item.text if hasattr(item, "text") else item).strip()
            log(f"DECODER_RESULT={text_out}")
            if text_out:
                log(f"R62 CHECKPOINT PASSED — 30S TELUGU STT ({decoder})")
                break
        except Exception as exc:
            msg = f"{decoder}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            log(msg)

    if not text_out:
        raise RuntimeError("30-second STT failed: " + " | ".join(errors))

    (WORK / "r62_asr_smoke.txt").write_text(text_out + "\n", encoding="utf-8")
    log("\nR62 DIAGNOSTIC COMPLETE — ASR INSTALL, MODEL RESTORE AND STT SMOKE PASSED")

except Exception as exc:
    log("\n" + "=" * 78)
    log("R62 DIAGNOSTIC FAILED")
    log("=" * 78)
    log(f"{type(exc).__name__}: {exc}")
    log(traceback.format_exc())
    log("The exception was captured intentionally so the kernel log can be retrieved by GitHub Actions.")
