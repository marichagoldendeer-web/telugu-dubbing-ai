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
    owner, name = slug.split("/", 1)
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
    log("R55 — ROBUST TELUGU STT + FULL TRANSLATION")
    log("=" * 72)

    log("\n[1] Dataset checkpoint")
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
    video = videos[0]
    nemo = nemos[0]
    log(f"VIDEO={video}")
    log(f"NEMO={nemo}")

    log("\n[2] GPU checkpoint")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    log(gpu.stdout.strip())
    if gpu.returncode != 0:
        raise RuntimeError("NVIDIA GPU not detected")

    log("\n[3] Install ASR runtime using AI4Bharat's recommended NeMo setup")
    nemo_repo = WORK / "NeMo"
    if not nemo_repo.exists():
        cmd(["git", "clone", "--depth", "1", "--branch", "nemo-v2",
             "https://github.com/AI4Bharat/NeMo.git", str(nemo_repo)])
    cmd(["bash", "reinstall.sh"], cwd=str(nemo_repo))
    cmd([sys.executable, "-m", "pip", "install", "-q",
         "numpy==1.26.4", "pyarrow<21.0.0"])
    log("NeMo environment installation checkpoint passed")

    log("\n[4] Install verified IndicTrans2 runtime stack")
    cmd([sys.executable, "-m", "pip", "install", "-q",
         "transformers==4.57.6", "huggingface_hub==0.36.2",
         "IndicTransToolkit", "sentencepiece", "sacremoses", "soundfile"])
    log("Translation dependency checkpoint passed")

    log("\n[5] Extract Telugu audio")
    wav = WORK / "r55_full_audio.wav"
    cmd(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)])
    if not wav.exists() or wav.stat().st_size < 1000:
        raise RuntimeError("Full audio extraction failed")
    log(f"WAV={wav} SIZE={wav.stat().st_size}")

    log("\n[6] Load IndicConformer")
    import torch
    import nemo.collections.asr as nemo_asr

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = "cuda"
    log(f"Torch={torch.__version__}")
    log(f"CUDA={torch.version.cuda}")

    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo), map_location=device
    )
    model.freeze()
    model = model.to(device)
    log(f"ASR_MODEL={type(model).__name__}")

    log("\n[7] Telugu STT")
    transcript_result = None
    errors = []
    for decoder in ["rnnt", "ctc"]:
        try:
            if hasattr(model, "cur_decoder"):
                model.cur_decoder = decoder
            log(f"Trying decoder={decoder}")
            try:
                result = model.transcribe(
                    [str(wav)], batch_size=1, logprobs=False, language_id="te"
                )
            except TypeError:
                result = model.transcribe([str(wav)], batch_size=1)
            item = result[0]
            text_value = item.text if hasattr(item, "text") else str(item)
            text_value = str(text_value).strip()
            if text_value:
                transcript_result = text_value
                log(f"STT succeeded with decoder={decoder}")
                break
        except Exception as exc:
            errors.append(f"{decoder}: {type(exc).__name__}: {exc}")
            log(errors[-1])

    if not transcript_result:
        raise RuntimeError("ASR failed: " + " | ".join(errors))

    transcript_path = WORK / "r55_telugu_transcript.txt"
    transcript_path.write_text(transcript_result + "\n", encoding="utf-8")
    log(f"TELUGU_TRANSCRIPT_CHARS={len(transcript_result)}")
    log(f"TELUGU_TRANSCRIPT={transcript_result}")

    log("\n[8] IndicTrans2 Telugu → English")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    tokenizer = AutoTokenizer.from_pretrained(str(trans_root), trust_remote_code=True)
    translator = AutoModelForSeq2SeqLM.from_pretrained(
        str(trans_root), trust_remote_code=True
    ).to(device).eval()
    processor = IndicProcessor(inference=True)

    # Keep the first version conservative: translate the complete ASR text
    # when it fits the model context; otherwise split by Telugu sentence punctuation.
    pieces = [p.strip() for p in transcript_result.replace("।", ".").split(".") if p.strip()]
    if not pieces:
        pieces = [transcript_result]

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
        translations.extend(processor.postprocess_batch(
            decoded, lang="eng_Latn"
        ))

    english = "\n".join(x.strip() for x in translations if x.strip())
    if not english:
        raise RuntimeError("IndicTrans2 produced an empty translation")

    english_path = WORK / "r55_english_translation.txt"
    english_path.write_text(english + "\n", encoding="utf-8")

    result = {
        "status": "PASSED",
        "video": str(video),
        "audio": str(wav),
        "asr_model": str(nemo),
        "transcript_file": str(transcript_path),
        "translation_file": str(english_path),
        "telugu_transcript_chars": len(transcript_result),
        "english_translation_chars": len(english),
    }
    (WORK / "r55_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log(f"ENGLISH_TRANSLATION={english}")
    log("\n" + "=" * 72)
    log("R55 CHECKPOINT PASSED — FULL TELUGU STT + TRANSLATION")
    log("=" * 72)
    save()

except Exception as exc:
    log("\n" + "=" * 72)
    log("R55 CHECKPOINT FAILED")
    log("=" * 72)
    log(f"{type(exc).__name__}: {exc}")
    tb = traceback.format_exc()
    print(tb, flush=True)
    lines.extend(tb.rstrip().splitlines())
    save()
    raise
