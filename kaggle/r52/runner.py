import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path("/kaggle/working/automation_result.txt")
WORK = Path("/kaggle/working")
NEMO_ROOT = WORK / "AI4Bharat-NeMo"
TRANSLATION_PKGS = Path("/tmp/indictrans_pkgs")
HF_CACHE = Path("/tmp/huggingface_cache")

R52_REQUIREMENTS = """huggingface-hub==0.36.2
transformers==4.57.6
indictranstoolkit==1.1.1
"""

SRC_LANG = "tel_Telu"
TGT_LANG = "eng_Latn"
TRANSLATION_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"
ASR_MODEL_NAME = "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large"

lines = []

def log(message=""):
    print(message, flush=True)
    lines.append(str(message))

def save():
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResult saved to {OUT}", flush=True)

def run_checked(cmd, label, cwd=None):
    log(f"\n{label}")
    log("$ " + " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
        lines.extend(result.stdout.rstrip().splitlines())
    if result.stderr:
        print(result.stderr.rstrip(), flush=True)
        lines.extend(result.stderr.rstrip().splitlines())
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    return result

def find_file(root, suffix):
    matches = list(root.rglob(f"*{suffix}"))
    return matches[0] if matches else None

try:
    log("=" * 70)
    log("R52 — REAL TELUGU AUDIO → STT → ENGLISH TRANSLATION")
    log("=" * 70)

    log("\n[1] Runtime")
    log(sys.version.splitlines()[0])
    log(platform.platform())

    log("\n[2] GPU")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if gpu.returncode != 0:
        raise RuntimeError("NVIDIA GPU not detected")
    log(gpu.stdout.strip())

    import torch
    log("\n[3] PyTorch / CUDA")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"CUDA version: {torch.version.cuda}")
    log(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    log("\n[4] Verify mounted datasets")

    video_root = Path("/kaggle/input/r52-telugu-source-video")
    indictrans_root = Path("/kaggle/input/indictrans2-indic-en-dist-200m")
    asr_root = Path("/kaggle/input/indicconformer-te-hybrid-rnnt-large")

    if not video_root.exists():
        raise FileNotFoundError(f"R52 source video dataset not mounted: {video_root}")
    if not indictrans_root.exists():
        raise FileNotFoundError(f"IndicTrans2 dataset not mounted: {indictrans_root}")
    if not asr_root.exists():
        raise FileNotFoundError(f"IndicConformer dataset not mounted: {asr_root}")

    video = find_file(video_root, ".mp4")
    asr_model = find_file(asr_root, ".nemo")

    if video is None:
        raise FileNotFoundError("No MP4 file found in R52 source video dataset")
    if asr_model is None:
        raise FileNotFoundError("No .nemo checkpoint found in IndicConformer dataset")

    log(f"Source video: {video}")
    log(f"IndicTrans2 path: {indictrans_root}")
    log(f"IndicConformer checkpoint: {asr_model}")
    log(f"IndicConformer size: {asr_model.stat().st_size} bytes")

    log("\n[5] Extract 16 kHz mono PCM audio")
    audio_dir = WORK / "r52"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio = audio_dir / "source_audio_16k_mono.wav"

    run_checked(
        [
            "ffmpeg", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(audio)
        ],
        "Extracting source audio"
    )

    if not audio.exists() or audio.stat().st_size < 10000:
        raise RuntimeError("Audio extraction produced an invalid WAV")

    log(f"Audio: {audio}")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio)
        ],
        capture_output=True, text=True
    )
    duration = float(probe.stdout.strip()) if probe.returncode == 0 else 0.0
    log(f"Audio duration: {duration:.2f} seconds")
    if duration < 1.0:
        raise RuntimeError("Extracted audio is too short")

    log("\n[6] Install AI4Bharat NeMo")
    if not NEMO_ROOT.exists():
        run_checked(
            [
                "git", "clone", "--depth", "1",
                "--branch", "nemo-v2",
                "https://github.com/AI4Bharat/NeMo.git",
                str(NEMO_ROOT)
            ],
            "Cloning AI4Bharat NeMo"
        )
    run_checked(
        ["bash", "reinstall.sh"],
        "Installing AI4Bharat NeMo dependencies",
        cwd=NEMO_ROOT
    )

    log("\n[7] Load IndicConformer RNNT checkpoint")
    from nemo.collections.asr.models import ASRModel

    asr_load_start = time.time()
    asr = ASRModel.restore_from(
        restore_path=str(asr_model),
        map_location="cuda"
    )
    asr.freeze()
    asr = asr.cuda()
    asr_load_seconds = time.time() - asr_load_start

    log(f"ASR model loaded in {asr_load_seconds:.2f} seconds")
    log(f"ASR model device: {next(asr.parameters()).device}")

    log("\n[8] Telugu RNNT transcription")
    asr.cur_decoder = "rnnt"
    stt_start = time.time()

    rnnt_result = asr.transcribe(
        [str(audio)],
        batch_size=1,
        language_id="te"
    )

    stt_seconds = time.time() - stt_start
    first = rnnt_result[0]

    if hasattr(first, "text"):
        telugu_text = first.text
    else:
        telugu_text = str(first)

    telugu_text = telugu_text.strip()

    log(f"RNNT inference time: {stt_seconds:.2f} seconds")
    log(f"Telugu transcript characters: {len(telugu_text)}")
    log("\nTELUGU TRANSCRIPT:")
    log(telugu_text)

    if len(telugu_text) < 10:
        raise RuntimeError("IndicConformer returned an unexpectedly short Telugu transcript")

    transcript_path = audio_dir / "telugu_rnnt_transcript.txt"
    transcript_path.write_text(telugu_text + "\n", encoding="utf-8")

    log(f"Transcript saved: {transcript_path}")

    log("\n[9] Install R49-compatible IndicTrans2 packages in /tmp")
    TRANSLATION_PKGS.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    req_file = Path("/tmp/r52-translation-requirements.txt")
    req_file.write_text(R52_REQUIREMENTS, encoding="utf-8")

    run_checked(
        [
            sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check", "--upgrade",
            "--target", str(TRANSLATION_PKGS),
            "-r", str(req_file)
        ],
        "Installing IndicTrans2 packages"
    )

    sys.path.insert(0, str(TRANSLATION_PKGS))
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")

    from IndicTransToolkit import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ip = IndicProcessor(inference=True)

    log("\n[10] Load IndicTrans2 from mounted dataset")
    tokenizer = AutoTokenizer.from_pretrained(
        str(indictrans_root),
        trust_remote_code=True,
        local_files_only=True
    )

    translation_model = AutoModelForSeq2SeqLM.from_pretrained(
        str(indictrans_root),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to("cuda")
    translation_model.eval()

    log(f"Translation model device: {next(translation_model.parameters()).device}")
    log(f"Translation model dtype: {next(translation_model.parameters()).dtype}")

    log("\n[11] Telugu → English translation")
    batch = ip.preprocess_batch(
        [telugu_text],
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG
    )

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True
    ).to("cuda")

    translation_start = time.time()

    with torch.no_grad():
        generated_tokens = translation_model.generate(
            **inputs,
            use_cache=False,
            max_length=512,
            num_beams=5,
            num_return_sequences=1
        )

    translation_seconds = time.time() - translation_start

    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    translations = ip.postprocess_batch(decoded, lang=TGT_LANG)

    if not translations or not translations[0].strip():
        raise RuntimeError("IndicTrans2 returned an empty English translation")

    english_text = translations[0].strip()

    log(f"Translation inference time: {translation_seconds:.2f} seconds")
    log("\nENGLISH TRANSLATION:")
    log(english_text)

    english_path = audio_dir / "english_translation.txt"
    english_path.write_text(english_text + "\n", encoding="utf-8")

    result_payload = {
        "checkpoint": "R52",
        "source_video": str(video),
        "audio": str(audio),
        "audio_duration_seconds": round(duration, 3),
        "asr_model": ASR_MODEL_NAME,
        "asr_decoder": "rnnt",
        "asr_language": "te",
        "asr_model_load_seconds": round(asr_load_seconds, 2),
        "asr_inference_seconds": round(stt_seconds, 2),
        "telugu_transcript_characters": len(telugu_text),
        "translation_model": TRANSLATION_MODEL,
        "translation_seconds": round(translation_seconds, 2),
        "english_translation_characters": len(english_text),
        "telugu_transcript_file": str(transcript_path),
        "english_translation_file": str(english_path)
    }

    metadata_path = audio_dir / "r52_result.json"
    metadata_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    log("\n[12] R52 result metadata")
    log(json.dumps(result_payload, ensure_ascii=False, indent=2))

    log("\n" + "=" * 70)
    log("R52 CHECKPOINT PASSED — REAL TELUGU STT + ENGLISH TRANSLATION")
    log("=" * 70)
    save()

except Exception as exc:
    log("\n" + "=" * 70)
    log("R52 CHECKPOINT FAILED")
    log("=" * 70)
    log(f"{type(exc).__name__}: {exc}")
    tb = traceback.format_exc()
    print(tb, flush=True)
    lines.extend(tb.rstrip().splitlines())
    save()
    raise
