import platform
import subprocess
import sys
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
LOG = WORK / "r63_diagnostic.txt"
VIDEO_DS = "anjiallkumar/r52-telugu-source-video"
ASR_DS = "anjiallkumar/indicconformer-te-hybrid-rnnt-large"
TRANSLATION_DS = "anjiallkumar/indictrans2-indic-en-dist-200m"

def log(msg=""):
    msg = str(msg)
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def run_cmd(args, cwd=None, timeout=None):
    log("$ " + " ".join(map(str, args)))
    try:
        p = subprocess.run(args, cwd=cwd, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, check=False)
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
    log("R63 — TELUGU STT + INDICTRANS2 TRANSLATION INTEGRATION")
    log("=" * 78)

    log("\n[1] RUNTIME / GPU")
    log(f"PYTHON={sys.version}")
    log(f"EXECUTABLE={sys.executable}")
    run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"])

    log("\n[2] DATASETS")
    video_root = locate(VIDEO_DS, "*.mp4")
    asr_root = locate(ASR_DS, "*.nemo")
    translation_root = locate(TRANSLATION_DS, "config.json")
    log(f"VIDEO_ROOT={video_root}")
    log(f"ASR_ROOT={asr_root}")
    log(f"TRANSLATION_ROOT={translation_root}")
    if not video_root or not asr_root or not translation_root:
        raise FileNotFoundError("One or more R53-verified datasets are missing")

    videos = sorted(video_root.rglob("*.mp4"))
    nemos = sorted(asr_root.rglob("*.nemo"))
    configs = sorted(translation_root.rglob("config.json"))
    if not videos or not nemos or not configs:
        raise FileNotFoundError("Required video, NeMo model, or IndicTrans config missing")

    video = videos[0]
    nemo_model_path = nemos[0]

    # Find the actual HF-style IndicTrans2 model directory.
    model_dirs = []
    for cfg in configs:
        parent = cfg.parent
        if ((parent / "tokenizer_config.json").exists() and
            ((parent / "model.safetensors").exists() or (parent / "pytorch_model.bin").exists())):
            model_dirs.append(parent)
    if not model_dirs:
        raise FileNotFoundError("Could not locate complete IndicTrans2 model directory")
    translation_model_dir = model_dirs[0]

    log(f"VIDEO={video}")
    log(f"NEMO_MODEL={nemo_model_path}")
    log(f"TRANSLATION_MODEL={translation_model_dir}")

    log("\n[3] NE MO SETUP")
    nemo_repo = WORK / "NeMo"
    if not nemo_repo.exists():
        rc = run_cmd(["git", "clone", "--depth", "1", "--branch", "nemo-v2",
                      "https://github.com/AI4Bharat/NeMo.git", str(nemo_repo)], timeout=300)
        if rc != 0:
            raise RuntimeError("NeMo clone failed")

    rc = run_cmd([sys.executable, "-m", "pip", "install",
                  "Cython", "packaging", "numpy==1.26.4", "pyarrow<21.0.0"], timeout=600)
    if rc != 0:
        raise RuntimeError("NeMo baseline dependency install failed")

    rc = run_cmd([sys.executable, "-m", "pip", "install", "-e", ".[asr]"],
                 cwd=str(nemo_repo), timeout=1200)
    if rc != 0:
        raise RuntimeError("NeMo ASR installation failed")

    rc = run_cmd([sys.executable, "-m", "pip", "install", "--force-reinstall",
                  "numpy==1.26.4", "pyarrow<21.0.0"], timeout=600)
    if rc != 0:
        raise RuntimeError("NeMo numpy/pyarrow restore failed")

    log("NEMO_INSTALL_PASSED")

    log("\n[4] TELUGU ASR")
    sys.path.insert(0, str(nemo_repo))
    import importlib.util
    log(f"NEMO_SPEC={importlib.util.find_spec('nemo')}")

    import torch
    log(f"TORCH={torch.__version__}")
    log(f"CUDA_AVAILABLE={torch.cuda.is_available()}")

    import pytorch_lightning.loggers as pl_loggers
    if not hasattr(pl_loggers, "NeptuneLogger"):
        class NeptuneLoggerCompat:
            pass
        pl_loggers.NeptuneLogger = NeptuneLoggerCompat
        log("NEPTUNELOGGER_SHIM=APPLIED")

    import nemo
    import nemo.collections.asr as nemo_asr
    log(f"NEMO_FILE={getattr(nemo, '__file__', None)}")
    log("NEMO_ASR_IMPORT_PASSED")

    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo_model_path), map_location="cuda"
    )
    log(f"MODEL_CLASS={type(model).__name__}")
    model.freeze()
    model = model.to("cuda")
    model.cur_decoder = "ctc"
    log("MODEL_RESTORE_PASSED")
    log("ASR_DECODER=ctc")
    try:
        lang_keys = list(model.ctc_decoder.language_masks.keys())
        log(f"CTC_LANGUAGE_KEYS={lang_keys}")
    except Exception as lang_exc:
        log(f"CTC_LANGUAGE_KEYS_UNAVAILABLE={type(lang_exc).__name__}: {lang_exc}")

    smoke = WORK / "r63_smoke_30s.wav"
    rc = run_cmd(["ffmpeg", "-y", "-ss", "0", "-t", "30", "-i", str(video),
                  "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(smoke)], timeout=180)
    if rc != 0 or not smoke.exists():
        raise RuntimeError("ffmpeg audio extraction failed")

    result = model.transcribe(
        [str(smoke)],
        batch_size=1,
        logprobs=False,
        language_id="te",
    )
    item = result[0]
    if hasattr(item, "text"):
        transcript = str(item.text)
    elif isinstance(item, (list, tuple)):
        transcript = " ".join(str(x) for x in item)
    else:
        transcript = str(item)
    transcript = transcript.strip()
    if transcript.startswith("[") and transcript.endswith("]"):
        # Defensive cleanup for the hybrid decoder's list-like return.
        import ast
        try:
            vals = ast.literal_eval(transcript)
            if isinstance(vals, list):
                transcript = " ".join(str(x) for x in vals).strip()
        except Exception:
            pass
    if not transcript:
        raise RuntimeError("Telugu STT returned empty text")
    log(f"TELUGU_TRANSCRIPT={transcript}")
    (WORK / "r63_telugu_transcript.txt").write_text(transcript + "\n", encoding="utf-8")
    log("R63 CHECKPOINT 1 PASSED — TELUGU STT")

    log("\n[5] INDIC TRANS2 DEPENDENCIES")
    # NeMo pins huggingface_hub low; IndicTrans2's current HF interface needs a
    # newer Hub. Upgrade only after ASR/model work is complete.
    rc = run_cmd([sys.executable, "-m", "pip", "install",
                  "--upgrade", "huggingface-hub==0.36.2"], timeout=300)
    if rc != 0:
        raise RuntimeError("huggingface-hub upgrade failed")

    rc = run_cmd([sys.executable, "-m", "pip", "install",
                  "--no-deps", "IndicTransToolkit"], timeout=300)
    if rc != 0:
        raise RuntimeError("IndicTransToolkit install failed")
    log("INDICTRANS_TOOLKIT_INSTALL_PASSED")

    log("\n[6] INDIC TRANS2 TRANSLATION")
    worker = WORK / "r63_translation_worker.py"
    worker.write_text(r'''
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

model_dir = Path(sys.argv[1])
text_file = Path(sys.argv[2])
out_file = Path(sys.argv[3])

src_lang = "tel_Telu"
tgt_lang = "eng_Latn"
text = text_file.read_text(encoding="utf-8").strip()

print(f"WORKER_MODEL={model_dir}", flush=True)
print(f"WORKER_TEXT={text}", flush=True)
import transformers
import huggingface_hub
print(f"TRANSFORMERS={transformers.__version__}", flush=True)
print(f"HUB={huggingface_hub.__version__}", flush=True)
print(f"TORCH={torch.__version__} CUDA={torch.cuda.is_available()}", flush=True)

ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained(
    str(model_dir), trust_remote_code=True, local_files_only=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    str(model_dir), trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
inputs = tokenizer(
    batch, truncation=True, padding="longest",
    max_length=256, return_tensors="pt"
).to(device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs, num_beams=5, num_return_sequences=1, max_length=256
    )

decoded = tokenizer.batch_decode(
    outputs.detach().cpu().tolist(),
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
translations = ip.postprocess_batch(decoded, lang=tgt_lang)
translation = str(translations[0]).strip()

print(f"ENGLISH_TRANSLATION={translation}", flush=True)
out_file.write_text(translation + "\n", encoding="utf-8")
''', encoding="utf-8")

    rc = run_cmd([sys.executable, str(worker), str(translation_model_dir),
                  str(WORK / "r63_telugu_transcript.txt"),
                  str(WORK / "r63_english_translation.txt")], timeout=900)
    if rc != 0:
        raise RuntimeError("IndicTrans2 translation worker failed")

    translation = (WORK / "r63_english_translation.txt").read_text(encoding="utf-8").strip()
    if not translation:
        raise RuntimeError("IndicTrans2 returned empty English translation")

    log(f"ENGLISH_TRANSLATION={translation}")
    log("R63 CHECKPOINT 2 PASSED — TELUGU → ENGLISH INDIC TRANS2")
    log("\nR63 DIAGNOSTIC COMPLETE — STT + INDIC TRANS2 TRANSLATION PASSED")

except Exception as exc:
    log("\n" + "=" * 78)
    log("R63 DIAGNOSTIC FAILED")
    log("=" * 78)
    log(f"{type(exc).__name__}: {exc}")
    log(traceback.format_exc())
    log("The exception was captured intentionally so GitHub Actions can retrieve the Kaggle log.")
