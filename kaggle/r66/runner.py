import asyncio
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
LOG = WORK / "r66_timed_dubbing.txt"
VIDEO_DS = "anjiallkumar/r52-telugu-source-video"
ASR_DS = "anjiallkumar/indicconformer-te-hybrid-rnnt-large"
TRANSLATION_DS = "anjiallkumar/indictrans2-indic-en-dist-200m"

CLIP_SECONDS = 30.0
VOICE = "en-US-GuyNeural"
TARGET_LANG = "eng_Latn"
SOURCE_LANG = "tel_Telu"

def log(msg=""):
    msg = str(msg)
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def run_cmd(args, cwd=None, timeout=600):
    log("$ " + " ".join(map(str, args)))
    try:
        p = subprocess.run(
            args, cwd=cwd, text=True, stdout=subprocess.PIPE,
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

def locate(slug, marker=None):
    name = slug.split("/", 1)[1]
    base = Path("/kaggle/input")
    for p in (base / name, base / slug):
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

def as_float(value):
    try:
        return float(value)
    except Exception:
        return None

def normalize_words(raw):
    out = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            word = item.get("word") or item.get("text") or item.get("token")
            start = as_float(item.get("start"))
            end = as_float(item.get("end"))
            if word is not None and start is not None and end is not None and end > start:
                out.append({"word": str(word).strip(), "start": start, "end": end})
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start = as_float(item[0])
            end = as_float(item[1])
            word = item[2]
            if start is not None and end is not None and word is not None and end > start:
                out.append({"word": str(word).strip(), "start": start, "end": end})
    return [x for x in out if x["word"]]

def normalize_segments(raw):
    out = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            start = as_float(item.get("start"))
            end = as_float(item.get("end"))
            text = item.get("text") or item.get("segment") or item.get("word")
            if start is not None and end is not None and end > start and text:
                out.append({"start": start, "end": end, "text": str(text).strip()})
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start = as_float(item[0])
            end = as_float(item[1])
            text = item[2]
            if start is not None and end is not None and end > start and text:
                out.append({"start": start, "end": end, "text": str(text).strip()})
    return out

def words_to_segments(words, gap_threshold=0.75, max_words=18):
    segments = []
    current = []
    for w in words:
        if current:
            gap = w["start"] - current[-1]["end"]
            if gap > gap_threshold or len(current) >= max_words:
                text = " ".join(x["word"] for x in current).strip()
                if text:
                    segments.append({
                        "start": current[0]["start"],
                        "end": current[-1]["end"],
                        "text": text
                    })
                current = []
        current.append(w)
    if current:
        text = " ".join(x["word"] for x in current).strip()
        if text:
            segments.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": text
            })
    return segments

def atempo_chain(speed):
    # speed > 1 means speed up. Split into safe factors for ffmpeg's atempo.
    parts = []
    remaining = float(speed)
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)

def media_duration(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    try:
        return float(p.stdout.strip())
    except Exception:
        return 0.0

async def synthesize(text, voice, out_path):
    import edge_tts
    await edge_tts.Communicate(text, voice).save(str(out_path))

try:
    LOG.write_text("", encoding="utf-8")
    log("=" * 78)
    log("R66-V2 — TIMED TELUGU → ENGLISH DUBBING SAMPLE")
    log("RUNNER_REVISION=R66-V2-CWD-VERIFIED")
    log("=" * 78)

    log("\n[1] RUNTIME / GPU")
    log(f"PYTHON={sys.version}")
    run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], timeout=120)

    log("\n[2] DATASETS")
    video_root = locate(VIDEO_DS, "*.mp4")
    asr_root = locate(ASR_DS, "*.nemo")
    translation_root = locate(TRANSLATION_DS, "config.json")
    log(f"VIDEO_ROOT={video_root}")
    log(f"ASR_ROOT={asr_root}")
    log(f"TRANSLATION_ROOT={translation_root}")
    if not video_root or not asr_root or not translation_root:
        raise FileNotFoundError("One or more required datasets are missing")

    video = sorted(video_root.rglob("*.mp4"))[0]
    nemo_model_path = sorted(asr_root.rglob("*.nemo"))[0]
    configs = sorted(translation_root.rglob("config.json"))
    model_dirs = []
    for cfg in configs:
        parent = cfg.parent
        if ((parent / "tokenizer_config.json").exists() and
            ((parent / "model.safetensors").exists() or (parent / "pytorch_model.bin").exists())):
            model_dirs.append(parent)
    if not model_dirs:
        raise FileNotFoundError("Complete IndicTrans2 model directory not found")
    translation_model_dir = model_dirs[0]

    log(f"VIDEO={video}")
    log(f"NEMO_MODEL={nemo_model_path}")
    log(f"TRANSLATION_MODEL={translation_model_dir}")

    log("\n[3] EXTRACT 30s AUDIO")
    smoke = WORK / "r66_source_30s.wav"
    rc = run_cmd([
        "ffmpeg", "-y", "-ss", "0", "-t", str(CLIP_SECONDS),
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(smoke)
    ], timeout=180)
    if rc != 0 or not smoke.exists():
        raise RuntimeError("30s audio extraction failed")

    log("\n[4] NE MO SETUP")
    nemo_repo = WORK / "NeMo"
    if not nemo_repo.exists():
        rc = run_cmd(["git", "clone", "--depth", "1", "--branch", "nemo-v2",
                       "https://github.com/AI4Bharat/NeMo.git", str(nemo_repo)], timeout=300)
        if rc != 0:
            raise RuntimeError("NeMo clone failed")

    rc = run_cmd([sys.executable, "-m", "pip", "install",
                  "Cython", "packaging", "numpy==1.26.4", "pyarrow<21.0.0"], timeout=600)
    if rc != 0:
        raise RuntimeError("NeMo baseline dependencies failed")

    log(f"NEMO_INSTALL_CWD={nemo_repo}")
    log(f"NEMO_SETUP_EXISTS={(nemo_repo / 'setup.py').exists()}")
    log(f"NEMO_PYPROJECT_EXISTS={(nemo_repo / 'pyproject.toml').exists()}")
    rc = run_cmd(
        [sys.executable, "-m", "pip", "install", "-e", ".[asr]"],
        cwd=str(nemo_repo),
        timeout=1200,
    )
    if rc != 0:
        raise RuntimeError("NeMo ASR installation failed")

    rc = run_cmd([sys.executable, "-m", "pip", "install", "--force-reinstall",
                  "numpy==1.26.4", "pyarrow<21.0.0"], timeout=600)
    if rc != 0:
        raise RuntimeError("NeMo dependency restore failed")

    sys.path.insert(0, str(nemo_repo))
    import torch
    import pytorch_lightning.loggers as pl_loggers
    if not hasattr(pl_loggers, "NeptuneLogger"):
        class NeptuneLoggerCompat:
            pass
        pl_loggers.NeptuneLogger = NeptuneLoggerCompat

    import nemo
    import nemo.collections.asr as nemo_asr
    log(f"TORCH={torch.__version__}")
    log(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
    log(f"NEMO_ACTIVE_CWD={Path.cwd()}")
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo_model_path), map_location="cuda"
    )
    model.freeze()
    model = model.to("cuda")
    model.cur_decoder = "ctc"
    log("MODEL_RESTORE_PASSED")

    log("\n[5] TIMESTAMPED TELUGU ASR")
    # This installed NeMo hybrid RNNT/CTC build exposes timestamp computation
    # through the decoder configuration rather than the transcribe() keyword.
    try:
        from omegaconf import open_dict
        with open_dict(model.cfg.decoding):
            model.cfg.decoding.compute_timestamps = True
        model.change_decoding_strategy(model.cfg.decoding, verbose=False)
        log("TIMESTAMP_DECODER_CONFIGURED=True")
    except Exception as exc:
        log(f"TIMESTAMP_DECODER_CONFIGURE_WARNING={type(exc).__name__}: {exc}")

    result = model.transcribe(
        [str(smoke)],
        batch_size=1,
        logprobs=False,
        language_id="te",
        return_hypotheses=True,
    )
    hyp = result[0]
    # Hybrid models may return a one-element nested hypothesis list.
    if isinstance(hyp, (list, tuple)) and len(hyp) == 1:
        hyp = hyp[0]
    log(f"HYPOTHESIS_TYPE={type(hyp).__name__}")

    segments = []
    for attr in ("timestamp", "timestep"):
        data = getattr(hyp, attr, None)
        if isinstance(data, dict):
            log(f"{attr.upper()}_KEYS={list(data.keys())}")
            segments = normalize_segments(data.get("segment"))
            if segments:
                break

    if not segments:
        words = []
        for attr in ("timestamp", "timestep"):
            data = getattr(hyp, attr, None)
            if isinstance(data, dict):
                words = normalize_words(data.get("word"))
                if words:
                    log(f"{attr.upper()}_WORD_COUNT={len(words)}")
                    break
        if words:
            segments = words_to_segments(words)

    if not segments:
        raise RuntimeError("NeMo returned no usable segment or word timestamps")

    clipped = []
    for s in segments:
        start = max(0.0, min(CLIP_SECONDS, s["start"]))
        end = max(0.0, min(CLIP_SECONDS, s["end"]))
        if end - start >= 0.30 and s["text"].strip():
            clipped.append({"start": start, "end": end, "text": s["text"].strip()})
    segments = clipped
    if not segments:
        raise RuntimeError("No valid speech segments inside 30s clip")

    log(f"ASR_SEGMENT_COUNT={len(segments)}")
    (WORK / "r66_asr_segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for i, s in enumerate(segments, 1):
        log(f"SEGMENT_{i}={s['start']:.3f}-{s['end']:.3f} | {s['text']}")

    log("R66 CHECKPOINT 1 PASSED — TIMESTAMPED TELUGU ASR")

    log("\n[6] INDIC TRANS2 TRANSLATION")
    rc = run_cmd([sys.executable, "-m", "pip", "install", "--upgrade",
                  "huggingface-hub==0.36.2"], timeout=300)
    if rc != 0:
        raise RuntimeError("huggingface-hub upgrade failed")
    rc = run_cmd([sys.executable, "-m", "pip", "install",
                  "sacremoses", "sacrebleu", "indic-nlp-library-itt"], timeout=300)
    if rc != 0:
        raise RuntimeError("IndicTrans runtime dependencies failed")
    rc = run_cmd([sys.executable, "-m", "pip", "install", "--no-deps",
                  "IndicTransToolkit"], timeout=300)
    if rc != 0:
        raise RuntimeError("IndicTransToolkit install failed")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    ip = IndicProcessor(inference=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(translation_model_dir), trust_remote_code=True, local_files_only=True
    )
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(
        str(translation_model_dir), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trans_model = trans_model.to(device)
    trans_model.eval()

    source_texts = [s["text"] for s in segments]
    batch = ip.preprocess_batch(source_texts, src_lang=SOURCE_LANG, tgt_lang=TARGET_LANG)
    inputs = tokenizer(
        batch, truncation=True, padding="longest", max_length=256, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = trans_model.generate(
            **inputs, num_beams=5, num_return_sequences=1, max_length=256
        )
    decoded = tokenizer.batch_decode(
        outputs.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    translations = ip.postprocess_batch(decoded, lang=TARGET_LANG)

    for s, translated in zip(segments, translations):
        s["english"] = str(translated).strip()
        if not s["english"]:
            raise RuntimeError("IndicTrans2 returned an empty segment translation")

    (WORK / "r66_segment_manifest.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for i, s in enumerate(segments, 1):
        log(f"TRANSLATION_{i}={s['english']}")

    log("R66 CHECKPOINT 2 PASSED — SEGMENT TRANSLATION")

    log("\n[7] SEGMENT TTS + TIMING FIT")
    rc = run_cmd([sys.executable, "-m", "pip", "install",
                  "--disable-pip-version-check", "--no-cache-dir",
                  "edge-tts==7.2.7"], timeout=300)
    if rc != 0:
        raise RuntimeError("edge-tts installation failed")
    import edge_tts

    positioned = []
    segment_meta = []
    for i, s in enumerate(segments, 1):
        raw = WORK / f"r66_tts_{i:02d}_raw.mp3"
        fit = WORK / f"r66_tts_{i:02d}_fit.wav"
        awaitable = synthesize(s["english"], VOICE, raw)
        asyncio.run(awaitable)
        actual = media_duration(raw)
        target = max(0.30, s["end"] - s["start"])
        if actual <= 0:
            raise RuntimeError(f"TTS duration unavailable for segment {i}")

        speed = max(1.0, actual / target)
        chain = atempo_chain(speed)
        pad_dur = max(0.0, target)
        af = f"{chain},apad=pad_dur={pad_dur:.6f},atrim=duration={target:.6f},asetpts=PTS-STARTPTS"

        rc = run_cmd([
            "ffmpeg", "-y", "-i", str(raw),
            "-af", af, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            str(fit)
        ], timeout=120)
        if rc != 0 or not fit.exists():
            raise RuntimeError(f"Timing fit failed for segment {i}")

        positioned.append((s["start"], fit))
        s["tts_duration_raw"] = actual
        s["target_duration"] = target
        s["speed_factor"] = speed
        s["tts_file"] = str(fit)
        segment_meta.append(s)
        log(f"TTS_{i}: raw={actual:.3f}s target={target:.3f}s speed={speed:.3f}")

    (WORK / "r66_segment_manifest.json").write_text(
        json.dumps(segment_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log("\n[8] BUILD TIMED ENGLISH DUB AUDIO")
    inputs = []
    filters = []
    for i, (start, fit) in enumerate(positioned):
        inputs += ["-i", str(fit)]
        delay = int(round(start * 1000))
        filters.append(f"[{i}:a]adelay={delay}:all=1[a{i}]")

    labels = "".join(f"[a{i}]" for i in range(len(positioned)))
    filters.append(
        f"{labels}amix=inputs={len(positioned)}:duration=longest:dropout_transition=0,"
        f"apad,atrim=duration={CLIP_SECONDS:.3f},asetpts=PTS-STARTPTS[out]"
    )
    dub_audio = WORK / "r66_english_dub_30s.wav"
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", ";".join(filters),
        "-map", "[out]", "-ar", "48000", "-ac", "1",
        "-c:a", "pcm_s16le", str(dub_audio)
    ]
    rc = run_cmd(cmd, timeout=240)
    if rc != 0 or not dub_audio.exists():
        raise RuntimeError("Timed English dub audio build failed")

    log("\n[9] MUX DUBBED VIDEO")
    dubbed_video = WORK / "r66_timed_dub_30s.mp4"
    rc = run_cmd([
        "ffmpeg", "-y", "-ss", "0", "-t", str(CLIP_SECONDS),
        "-i", str(video), "-i", str(dub_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(dubbed_video)
    ], timeout=240)
    if rc != 0 or not dubbed_video.exists():
        raise RuntimeError("Dubbed video mux failed")

    rc = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,format_name",
        "-of", "default=noprint_wrappers=1",
        str(dubbed_video)
    ], timeout=120)
    if rc != 0:
        raise RuntimeError("Dubbed video validation failed")

    log("R66 CHECKPOINT 3 PASSED — TIMED ENGLISH TTS + AUDIO BUILD")
    log("R66 CHECKPOINT 4 PASSED — 30s DUBBED VIDEO")
    log("\nR66 DIAGNOSTIC COMPLETE — TIMED TELUGU → ENGLISH DUBBING PASSED")

except Exception as exc:
    log("\n" + "=" * 78)
    log("R66 DIAGNOSTIC FAILED")
    log("=" * 78)
    log(f"{type(exc).__name__}: {exc}")
    log(traceback.format_exc())
    raise
