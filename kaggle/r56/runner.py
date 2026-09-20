import json
import subprocess
import sys
import traceback
from pathlib import Path

OUT = Path("/kaggle/working/automation_result.txt")
WORK = Path("/kaggle/working")
VIDEO_DS = "anjiallkumar/r52-telugu-source-video"
TRANS_DS = "anjiallkumar/indictrans2-indic-en-dist-200m"
ASR_DS = "anjiallkumar/indicconformer-te-hybrid-rnnt-large"

lines = []

def log(x=""):
    print(x, flush=True)
    lines.append(str(x))

def save():
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

def locate(slug, marker=None):
    name = slug.split("/", 1)[1]
    for p in [Path("/kaggle/input") / name, Path("/kaggle/input") / slug]:
        if p.exists():
            return p
    base = Path("/kaggle/input")
    if not base.exists():
        return None
    matches = [p for p in base.rglob("*") if p.is_dir() and p.name == name]
    if matches:
        return matches[0]
    if marker:
        matches = [p.parent for p in base.rglob(marker) if p.is_file()]
        if matches:
            return matches[0]
    return None

def cmd(args, cwd=None, check=True):
    log("$ " + " ".join(map(str, args)))
    return subprocess.run(args, cwd=cwd, check=check, text=True)

try:
    log("=" * 72)
    log("R56 — LEAN NEVOCONFORMER ASR + TRANSLATION")
    log("=" * 72)

    log("\n[1] DATASET CHECKPOINT")
    video_root = locate(VIDEO_DS, "*.mp4")
    trans_root = locate(TRANS_DS, "model.safetensors")
    asr_root = locate(ASR_DS, "*.nemo")
    log(f"VIDEO_ROOT={video_root}")
    log(f"TRANS_ROOT={trans_root}")
    log(f"ASR_ROOT={asr_root}")
    if not video_root or not trans_root or not asr_root:
        raise FileNotFoundError("R53-verified dataset root missing")

    videos = sorted(video_root.rglob("*.mp4"))
    nemos = sorted(asr_root.rglob("*.nemo"))
    if not videos or not nemos:
        raise FileNotFoundError("Required MP4 or .nemo file missing")
    video, nemo = videos[0], nemos[0]
    log(f"VIDEO={video}")
    log(f"NEMO={nemo}")

    log("\n[2] GPU CHECKPOINT")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    log(gpu.stdout.strip())
    if gpu.returncode != 0:
        raise RuntimeError("NVIDIA GPU not detected")

    log("\n[3] LEAN AI4BHARAT NEMO INSTALL")
    nemo_repo = WORK / "NeMo"
    if not nemo_repo.exists():
        cmd(["git", "clone", "--depth", "1", "--branch", "nemo-v2",
             "https://github.com/AI4Bharat/NeMo.git", str(nemo_repo)])

    cmd([sys.executable, "-m", "pip", "install", "-q",
         "Cython", "packaging", "numpy==1.26.4", "pyarrow<21.0.0"])
    cmd([sys.executable, "-m", "pip", "install", "-q", "-e", ".[asr]"], cwd=str(nemo_repo))
    log("LEAN_NEMO_INSTALL_PASSED")

    log("\n[4] ASR IMPORT CHECK")
    import torch
    import nemo.collections.asr as nemo_asr
    log(f"TORCH={torch.__version__}")
    log(f"CUDA={torch.version.cuda}")
    log(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after NeMo installation")
    log(f"NEMO_ASR_IMPORT={nemo_asr.__name__}")

    log("\n[5] EXTRACT 30-SECOND SMOKE AUDIO")
    smoke_wav = WORK / "r56_smoke_30s.wav"
    cmd(["ffmpeg", "-y", "-ss", "0", "-t", "30", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(smoke_wav)])
    if not smoke_wav.exists() or smoke_wav.stat().st_size < 1000:
        raise RuntimeError("30-second audio extraction failed")
    log(f"SMOKE_WAV={smoke_wav} SIZE={smoke_wav.stat().st_size}")

    log("\n[6] LOAD INDICCONFORMER")
    device = "cuda"
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo), map_location=device
    )
    model.freeze()
    model = model.to(device)
    log(f"ASR_MODEL={type(model).__name__}")

    log("\n[7] 30-SECOND TELUGU STT SMOKE TEST")
    smoke_text = None
    smoke_errors = []
    for decoder in ["ctc", "rnnt"]:
        try:
            if hasattr(model, "cur_decoder"):
                model.cur_decoder = decoder
            log(f"Trying smoke decoder={decoder}")
            try:
                result = model.transcribe(
                    [str(smoke_wav)], batch_size=1, logprobs=False, language_id="te"
                )
            except TypeError:
                result = model.transcribe([str(smoke_wav)], batch_size=1)
            item = result[0]
            value = item.text if hasattr(item, "text") else str(item)
            value = str(value).strip()
            if value:
                smoke_text = value
                log(f"ASR_SMOKE_PASSED decoder={decoder}")
                log(f"ASR_SMOKE_TEXT={value}")
                break
        except Exception as exc:
            msg = f"{decoder}: {type(exc).__name__}: {exc}"
            smoke_errors.append(msg)
            log(msg)

    if not smoke_text:
        raise RuntimeError("30-second ASR smoke test failed: " + " | ".join(smoke_errors))

    log("\nR56 CHECKPOINT 1 PASSED — LEAN NEMO + 30S TELUGU STT")

    log("\n[8] EXTRACT FULL AUDIO")
    full_wav = WORK / "r56_full_audio.wav"
    cmd(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(full_wav)])
    if not full_wav.exists() or full_wav.stat().st_size < 1000:
        raise RuntimeError("Full audio extraction failed")
    log(f"FULL_WAV={full_wav} SIZE={full_wav.stat().st_size}")

    log("\n[9] FULL TELUGU STT")
    transcript = None
    errors = []
    for decoder in ["ctc", "rnnt"]:
        try:
            if hasattr(model, "cur_decoder"):
                model.cur_decoder = decoder
            log(f"Trying full decoder={decoder}")
            try:
                result = model.transcribe(
                    [str(full_wav)], batch_size=1, logprobs=False, language_id="te"
                )
            except TypeError:
                result = model.transcribe([str(full_wav)], batch_size=1)
            item = result[0]
            value = item.text if hasattr(item, "text") else str(item)
            value = str(value).strip()
            if value:
                transcript = value
                log(f"FULL_STT_PASSED decoder={decoder}")
                break
        except Exception as exc:
            msg = f"{decoder}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            log(msg)
    if not transcript:
        raise RuntimeError("Full ASR failed: " + " | ".join(errors))

    transcript_path = WORK / "r56_telugu_transcript.txt"
    transcript_path.write_text(transcript + "\n", encoding="utf-8")
    log(f"TELUGU_TRANSCRIPT_CHARS={len(transcript)}")

    log("\n[10] INSTALL TRANSLATION STACK")
    cmd([sys.executable, "-m", "pip", "install", "-q",
         "transformers==4.57.6", "huggingface_hub==0.36.2",
         "IndicTransToolkit", "sentencepiece", "sacremoses", "soundfile"])
    log("TRANSLATION_DEPENDENCIES_PASSED")

    log("\n[11] INDICTRANS2 TELUGU -> ENGLISH")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    tokenizer = AutoTokenizer.from_pretrained(str(trans_root), trust_remote_code=True)
    translator = AutoModelForSeq2SeqLM.from_pretrained(
        str(trans_root), trust_remote_code=True
    ).to(device).eval()
    processor = IndicProcessor(inference=True)

    pieces = [p.strip() for p in transcript.replace("।", ".").split(".") if p.strip()]
    if not pieces:
        pieces = [transcript]

    translations = []
    for start in range(0, len(pieces), 8):
        chunk = pieces[start:start+8]
        batch = processor.preprocess_batch(
            chunk, src_lang="tel_Telu", tgt_lang="eng_Latn", visualize=False
        )
        encoded = tokenizer(
            batch, padding="longest", truncation=True, max_length=256,
            return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            generated = translator.generate(
                **encoded, num_beams=5, num_return_sequences=1, max_length=256
            )
        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        translations.extend(processor.postprocess_batch(decoded, lang="eng_Latn"))

    english = "\n".join(x.strip() for x in translations if x.strip())
    if not english:
        raise RuntimeError("IndicTrans2 produced an empty translation")

    english_path = WORK / "r56_english_translation.txt"
    english_path.write_text(english + "\n", encoding="utf-8")
    result = {
        "status": "PASSED",
        "video": str(video),
        "audio": str(full_wav),
        "asr_model": str(nemo),
        "transcript_file": str(transcript_path),
        "translation_file": str(english_path),
        "smoke_decoder": "passed",
        "telugu_transcript_chars": len(transcript),
        "english_translation_chars": len(english)
    }
    (WORK / "r56_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"ENGLISH_TRANSLATION_CHARS={len(english)}")
    log("\n" + "=" * 72)
    log("R56 CHECKPOINT PASSED — FULL TELUGU STT + TRANSLATION")
    log("=" * 72)
    save()

except Exception as exc:
    log("\n" + "=" * 72)
    log("R56 CHECKPOINT FAILED")
    log("=" * 72)
    log(f"{type(exc).__name__}: {exc}")
    tb = traceback.format_exc()
    print(tb, flush=True)
    lines.extend(tb.rstrip().splitlines())
    save()
    raise
