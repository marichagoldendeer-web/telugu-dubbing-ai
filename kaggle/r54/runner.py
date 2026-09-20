import json
import os
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
    candidates = [
        Path("/kaggle/input") / name,
        Path("/kaggle/input") / slug,
    ]
    for p in candidates:
        if p.exists():
            return p

    base = Path("/kaggle/input")
    if not base.exists():
        return None

    matches = [p for p in base.rglob("*") if p.is_dir() and p.name == name]
    if matches:
        return matches[0]

    if marker:
        marker_matches = [p.parent for p in base.rglob(marker) if p.is_file()]
        if marker_matches:
            return marker_matches[0]
    return None

def run(cmd, **kwargs):
    log("$ " + " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, text=True, **kwargs)

def install_runtime():
    log("\n[1] Installing R54 runtime dependencies")
    run([sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.51", "IndicTransToolkit", "sentencepiece",
         "sacremoses", "soundfile"])
    nemo_dir = WORK / "AI4Bharat-NeMo"
    if not nemo_dir.exists():
        run(["git", "clone", "--depth", "1", "--branch", "nemo-v2",
             "https://github.com/AI4Bharat/NeMo.git", str(nemo_dir)])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[asr]"],
        cwd=str(nemo_dir))

def find_video(root):
    vids = sorted(root.rglob("*.mp4"))
    if not vids:
        raise FileNotFoundError("No MP4 video found")
    return vids[0]

def find_nemo(root):
    models = sorted(root.rglob("*.nemo"))
    if not models:
        raise FileNotFoundError("No .nemo ASR checkpoint found")
    return models[0]

try:
    log("=" * 72)
    log("R54 — TELUGU STT + TELUGU→ENGLISH TRANSLATION")
    log("=" * 72)

    log("\n[1] Locating verified R53 datasets")
    video_root = locate(VIDEO_DS, "*.mp4")
    trans_root = locate(TRANS_DS, "model.safetensors")
    asr_root = locate(ASR_DS, "*.nemo")
    log(f"VIDEO_ROOT: {video_root}")
    log(f"TRANS_ROOT: {trans_root}")
    log(f"ASR_ROOT: {asr_root}")

    if not video_root or not trans_root or not asr_root:
        raise FileNotFoundError("One or more R53-verified datasets cannot be located")

    video = find_video(video_root)
    nemo_path = find_nemo(asr_root)
    log(f"VIDEO: {video}")
    log(f"ASR MODEL: {nemo_path}")

    log("\n[2] GPU")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    log(gpu.stdout.strip())
    if gpu.returncode != 0:
        raise RuntimeError("NVIDIA GPU not detected")

    install_runtime()

    log("\n[3] Extracting a 30-second Telugu audio smoke-test clip")
    wav = WORK / "r54_test_audio.wav"
    run([
        "ffmpeg", "-y", "-i", str(video), "-t", "30",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if not wav.exists() or wav.stat().st_size < 1000:
        raise RuntimeError("Audio extraction failed or produced an empty WAV")
    log(f"WAV: {wav} ({wav.stat().st_size} bytes)")

    log("\n[4] Loading AI4Bharat IndicConformer")
    import torch
    import nemo.collections.asr as nemo_asr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Torch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if device != "cuda":
        raise RuntimeError("R54 requires CUDA for the ASR smoke test")

    asr_model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo_path), map_location=device
    )
    asr_model.freeze()
    asr_model = asr_model.to(device)

    transcript = None
    decoder_attempts = []
    if hasattr(asr_model, "cur_decoder"):
        decoder_attempts = ["rnnt", "ctc"]
    else:
        decoder_attempts = [None]

    last_error = None
    for decoder in decoder_attempts:
        try:
            if decoder is not None:
                asr_model.cur_decoder = decoder
                log(f"Trying ASR decoder: {decoder}")
            try:
                result = asr_model.transcribe(
                    [str(wav)], batch_size=1, logprobs=False, language_id="te"
                )
            except TypeError:
                result = asr_model.transcribe([str(wav)], batch_size=1)

            item = result[0]
            if hasattr(item, "text"):
                transcript = item.text
            else:
                transcript = str(item)
            transcript = transcript.strip()
            if transcript:
                log(f"ASR decoder succeeded: {decoder or 'default'}")
                break
        except Exception as exc:
            last_error = exc
            log(f"Decoder {decoder or 'default'} failed: {type(exc).__name__}: {exc}")

    if not transcript:
        raise RuntimeError(f"ASR produced no transcript. Last error: {last_error}")

    transcript_file = WORK / "r54_telugu_transcript.txt"
    transcript_file.write_text(transcript + "\n", encoding="utf-8")
    log(f"TELUGU TRANSCRIPT: {transcript}")

    log("\n[5] Loading local IndicTrans2 Telugu→English model")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    tok = AutoTokenizer.from_pretrained(
        str(trans_root), trust_remote_code=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(trans_root), trust_remote_code=True
    ).to(device).eval()
    processor = IndicProcessor(inference=True)

    batch = processor.preprocess_batch(
        [transcript], src_lang="tel_Telu", tgt_lang="eng_Latn", visualize=False
    )
    encoded = tok(
        batch, padding="longest", truncation=True, max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        generated = model.generate(
            **encoded, num_beams=5, num_return_sequences=1, max_length=256
        )

    translated = tok.batch_decode(
        generated, skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    translated = processor.postprocess_batch(
        translated, lang="eng_Latn"
    )[0].strip()

    translation_file = WORK / "r54_english_translation.txt"
    translation_file.write_text(translated + "\n", encoding="utf-8")

    result_json = WORK / "r54_result.json"
    result_json.write_text(json.dumps({
        "video": str(video),
        "audio_smoke_test_seconds": 30,
        "asr_model": str(nemo_path),
        "transcript_telugu": transcript,
        "translation_english": translated,
        "status": "PASSED"
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"ENGLISH TRANSLATION: {translated}")
    log("\n" + "=" * 72)
    log("R54 CHECKPOINT PASSED — TELUGU STT + TRANSLATION")
    log("=" * 72)
    save()

except Exception as exc:
    log("\n" + "=" * 72)
    log("R54 CHECKPOINT FAILED")
    log("=" * 72)
    log(f"{type(exc).__name__}: {exc}")
    tb = traceback.format_exc()
    print(tb, flush=True)
    lines.extend(tb.rstrip().splitlines())
    save()
    raise
