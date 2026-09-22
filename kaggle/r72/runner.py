import asyncio
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
LOG = WORK / "r72_timed_dubbing.txt"
VIDEO_DS = "anjiallkumar/r52-telugu-source-video"
ASR_DS = "anjiallkumar/indicconformer-te-hybrid-rnnt-large"
TRANSLATION_DS = "anjiallkumar/indictrans2-indic-en-dist-200m"

CLIP_SECONDS = 30.0
VOICE = "en-IN-PrabhatNeural"
MAX_TTS_SPEED = 1.20
WORD_GAP_THRESHOLD = 0.40
MAX_WORDS_PER_SEGMENT = 8
MAX_SEGMENT_SECONDS = 3.20
MIN_SEGMENT_SECONDS = 0.70
OVERLAP_MIN_WORDS = 3
OVERLAP_MAX_WORDS = 8
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

def normalize_words(raw, time_stride=1.0):
    out = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            word = item.get("word") or item.get("text") or item.get("token") or item.get("char")
            start = as_float(item.get("start"))
            end = as_float(item.get("end"))
            if start is None:
                start = as_float(item.get("start_offset"))
            if end is None:
                end = as_float(item.get("end_offset"))
            if start is not None and end is not None:
                start *= time_stride
                end *= time_stride
            if word is not None and start is not None and end is not None and end > start:
                out.append({"word": str(word).strip(), "start": start, "end": end})
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start = as_float(item[0])
            end = as_float(item[1])
            word = item[2]
            if start is not None and end is not None and word is not None and end > start:
                out.append({"word": str(word).strip(), "start": start * time_stride, "end": end * time_stride})
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

def words_to_segments(words, gap_threshold=0.65, max_words=40):
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
    log("R72 — TIMED TELUGU → ENGLISH DUBBING WITH MUSIC BED")
    log("RUNNER_REVISION=R72-MUSIC-BED-CENTER-SUPPRESSION")
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

    log("\n[3] EXTRACT 30s AUDIO + ORIGINAL MUSIC BED")
    smoke = WORK / "r72_source_30s.wav"
    source_stereo = WORK / "r72_source_stereo_30s.wav"
    music_bed = WORK / "r72_music_bed_30s.wav"

    rc = run_cmd([
        "ffmpeg", "-y", "-ss", "0", "-t", str(CLIP_SECONDS),
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(smoke)
    ], timeout=180)
    if rc != 0 or not smoke.exists():
        raise RuntimeError("30s ASR audio extraction failed")

    # Keep the original stereo mix and derive a center-suppressed music/ambience
    # bed. Typical dialogue is center-panned; stereo music and ambience survive.
    rc = run_cmd([
        "ffmpeg", "-y", "-ss", "0", "-t", str(CLIP_SECONDS),
        "-i", str(video), "-vn", "-ac", "2", "-ar", "48000",
        "-c:a", "pcm_s16le", str(source_stereo)
    ], timeout=180)
    if rc != 0 or not source_stereo.exists():
        raise RuntimeError("30s stereo source extraction failed")

    rc = run_cmd([
        "ffmpeg", "-y", "-i", str(source_stereo),
        "-af", "pan=stereo|c0=0.5*FL-0.5*FR|c1=0.5*FR-0.5*FL,highpass=f=50,lowpass=f=18000,volume=1.6",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(music_bed)
    ], timeout=180)
    if rc != 0 or not music_bed.exists():
        raise RuntimeError("Music bed extraction failed")
    log("MUSIC_BED_METHOD=center-suppressed-stereo")

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
            # This fork requires alignment preservation before it will emit
            # character/word/segment timesteps.
            model.cfg.decoding.compute_timestamps = True
            model.cfg.decoding.compute_timesteps = True
            model.cfg.decoding.preserve_alignments = True
        # The AI4Bharat nemo-v2 fork exposes change_decoding_strategy() without
        # the newer verbose keyword. Keep the decoder configuration explicit.
        model.change_decoding_strategy(model.cfg.decoding)

        # change_decoding_strategy() rebuilds the decoding object, so the
        # multilingual tokenizer compatibility patch must be applied AFTER
        # that call. The timestamp helper invokes these methods without lang.
        try:
            import types
            decoding = model.decoding
            original_decode_tokens_to_str = decoding.decode_tokens_to_str
            original_decode_ids_to_tokens = decoding.decode_ids_to_tokens

            def decode_tokens_to_str_te(self, tokens, lang=None):
                return original_decode_tokens_to_str(tokens, "te" if lang is None else lang)

            def decode_ids_to_tokens_te(self, tokens, lang=None):
                return original_decode_ids_to_tokens(tokens, "te" if lang is None else lang)

            decoding.decode_tokens_to_str = types.MethodType(decode_tokens_to_str_te, decoding)
            decoding.decode_ids_to_tokens = types.MethodType(decode_ids_to_tokens_te, decoding)
            log("MULTILINGUAL_TOKENIZER_LANG_PATCH=True lang=te")
        except Exception as exc:
            log(f"MULTILINGUAL_TOKENIZER_LANG_PATCH_WARNING={type(exc).__name__}: {exc}")

        log(f"TIMESTAMP_DECODER_CONFIGURED=True compute_timestamps={model.cfg.decoding.get('compute_timestamps', None)} compute_timesteps={model.cfg.decoding.get('compute_timesteps', None)} preserve_alignments={model.cfg.decoding.get('preserve_alignments', None)}")
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
    timestamp_stride = 1.0
    try:
        subsampling = float(getattr(model.encoder, "subsampling_factor", 1) or 1)
        window_stride = float(model.cfg.preprocessor.window_stride)
        timestamp_stride = subsampling * window_stride
        log(f"TIMESTAMP_STRIDE={timestamp_stride:.6f}s subsampling_factor={subsampling:g} window_stride={window_stride:g}")
    except Exception as exc:
        log(f"TIMESTAMP_STRIDE_WARNING={type(exc).__name__}: {exc}")

    # R72: prefer word timestamps whenever NeMo provides them.
    # R68 proved that 24 word timestamps were available, but its segment-level
    # timestamp collapsed all speech into one 10.48s block. Smaller word-based
    # chunks give translation and TTS much better timing boundaries.
    words = []
    for attr in ("timestep", "timestamp"):
        data = getattr(hyp, attr, None)
        if isinstance(data, dict):
            log(f"{attr.upper()}_KEYS={list(data.keys())}")
            raw_words = data.get("word")
            if isinstance(raw_words, list) and raw_words:
                log(f"{attr.upper()}_WORD_SAMPLE={raw_words[0]}")
            words = normalize_words(raw_words, time_stride=timestamp_stride)
            if words:
                log(f"{attr.upper()}_WORD_COUNT={len(words)}")
                break

    if words:
        segments = words_to_segments(
            words,
            gap_threshold=WORD_GAP_THRESHOLD,
            max_words=MAX_WORDS_PER_SEGMENT,
        )
        # Split oversized word groups by duration as well.
        refined = []
        for s in segments:
            group_words = [w for w in words if w["start"] >= s["start"] - 1e-6 and w["end"] <= s["end"] + 1e-6]
            current = []
            for w in group_words:
                if current:
                    candidate_end = w["end"]
                    candidate_words = len(current) + 1
                    if (
                        candidate_words > MAX_WORDS_PER_SEGMENT
                        or candidate_end - current[0]["start"] > MAX_SEGMENT_SECONDS
                    ):
                        refined.append({
                            "start": current[0]["start"],
                            "end": current[-1]["end"],
                            "text": " ".join(x["word"] for x in current).strip(),
                        })
                        current = []
                current.append(w)
            if current:
                refined.append({
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "text": " ".join(x["word"] for x in current).strip(),
                })
        segments = refined
        log(f"SEGMENT_SOURCE=WORD_TIMESTAMPS")
    else:
        for attr in ("timestep", "timestamp"):
            data = getattr(hyp, attr, None)
            if isinstance(data, dict):
                candidate = normalize_segments(data.get("segment"))
                if candidate:
                    segments = candidate
                    log(f"SEGMENT_SOURCE={attr.upper()}_SEGMENTS_FALLBACK")
                    break

    if not segments:
        raise RuntimeError("NeMo returned no usable segment or word timestamps")

    clipped = []
    for s in segments:
        start = max(0.0, min(CLIP_SECONDS, s["start"]))
        end = max(0.0, min(CLIP_SECONDS, s["end"]))
        if end - start >= MIN_SEGMENT_SECONDS and s["text"].strip():
            clipped.append({"start": start, "end": end, "text": s["text"].strip()})
    segments = clipped

    # Only merge very small adjacent fragments; do not re-collapse normal
    # speech chunks the way R68's 0.35s merge could.
    merged = []
    for s in segments:
        if (
            merged
            and (s["start"] - merged[-1]["end"]) <= 0.15
            and (merged[-1]["end"] - merged[-1]["start"]) < MIN_SEGMENT_SECONDS
        ):
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))
    segments = merged

    # R72: remove repeated ASR phrase overlap between adjacent segments.
    # NeMo's word timestamps can repeat a phrase at a boundary. Keep the
    # first occurrence and remove the repeated phrase from the later segment.
    def _tokens(text):
        return [t for t in re.findall(r"\S+", text.strip()) if t]

    def _dedupe_adjacent(a, b):
        at = _tokens(a)
        bt = _tokens(b)
        best = None
        max_n = min(OVERLAP_MAX_WORDS, len(at), len(bt))
        for n in range(max_n, OVERLAP_MIN_WORDS - 1, -1):
            for ai in range(0, len(at) - n + 1):
                needle = at[ai:ai+n]
                for bi in range(0, len(bt) - n + 1):
                    if bt[bi:bi+n] == needle:
                        best = (ai, bi, n)
                        break
                if best:
                    break
            if best:
                break
        if not best:
            return a, b, None
        ai, bi, n = best
        # Prefer preserving unique words on both sides of the later segment.
        new_bt = bt[:bi] + bt[bi+n:]
        return " ".join(at), " ".join(new_bt), " ".join(bt[bi:bi+n])

    deduped = [dict(s) for s in segments]
    for i in range(1, len(deduped)):
        prev_text = deduped[i-1]["text"]
        cur_text = deduped[i]["text"]
        new_prev, new_cur, overlap = _dedupe_adjacent(prev_text, cur_text)
        if overlap:
            # If the overlap starts inside the previous segment, keep the
            # previous segment's original words and only trim the later copy.
            deduped[i]["text"] = new_cur.strip()
            log(f"R72_OVERLAP_REMOVED_{i}={overlap}")
    deduped = [s for s in deduped if s["text"].strip()]
    segments = deduped
    # R72: merge tiny orphan fragments into the following segment.
    # After overlap removal, a boundary word such as "సనాతన" can be left
    # alone. Translating it separately harms sentence quality and wastes its
    # timing window. Keep its original start and extend through the next
    # segment so translation receives the complete phrase.
    ORPHAN_MAX_WORDS = 2
    merged_orphans = []
    i = 0
    while i < len(segments):
        cur = dict(segments[i])
        cur_words = _tokens(cur["text"])
        if (
            len(cur_words) <= ORPHAN_MAX_WORDS
            and i + 1 < len(segments)
            and (segments[i + 1]["start"] - cur["end"]) <= 1.0
        ):
            nxt = dict(segments[i + 1])
            cur["end"] = nxt["end"]
            cur["text"] = (cur["text"] + " " + nxt["text"]).strip()
            log(f"R72_ORPHAN_MERGED_{i+1}={segments[i]['text']} + {nxt['text']}")
            merged_orphans.append(cur)
            i += 2
        else:
            merged_orphans.append(cur)
            i += 1
    segments = merged_orphans
    log(f"R72_SEGMENT_COUNT_AFTER_ORPHAN_MERGE={len(segments)}")
    log(f"R72_SEGMENT_COUNT_AFTER_OVERLAP_CLEAN={len(segments)}")
    log(f"ASR_SEGMENT_COUNT_AFTER_MERGE={len(segments)}")
    if not segments:
        raise RuntimeError("No valid speech segments inside 30s clip")

    log(f"ASR_SEGMENT_COUNT={len(segments)}")
    (WORK / "r72_asr_segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for i, s in enumerate(segments, 1):
        log(f"SEGMENT_{i}={s['start']:.3f}-{s['end']:.3f} | {s['text']}")

    log("R72 CHECKPOINT 1 PASSED — TIMESTAMPED TELUGU ASR + QUALITY SEGMENT MERGE")

    log("\n[6] INDIC TRANS2 TRANSLATION")
    # NeMo pins an older Transformers/Hub pair during ASR installation.
    # The IndicTrans2 custom model was already validated with this pair in R49,
    # so restore that known-good translation stack after ASR is complete.
    rc = run_cmd([sys.executable, "-m", "pip", "install", "--upgrade",
                  "transformers==4.57.6", "huggingface-hub==0.36.2"], timeout=600)
    if rc != 0:
        raise RuntimeError("IndicTrans Hugging Face stack upgrade failed")

    # Some Kaggle images expose the torch-sharding helper only through the
    # serialization submodule. Transformers imports it from the hub root.
    # Bridge that packaging difference without changing the pinned versions.
    try:
        # NeMo imports an older Transformers/Hub stack in this same Python
        # process. Pip replaces the files on disk, but already-imported modules
        # remain in sys.modules. Clear both package trees before loading the
        # translation stack so we actually use the newly installed versions.
        for module_name in list(sys.modules):
            if (
                module_name == "transformers"
                or module_name.startswith("transformers.")
                or module_name == "huggingface_hub"
                or module_name.startswith("huggingface_hub.")
            ):
                del sys.modules[module_name]

        import huggingface_hub
        import huggingface_hub.constants as hf_constants

        compat_constants = {
            "PYTORCH_WEIGHTS_FILE_PATTERN": "pytorch_model{suffix}.bin",
            "SAFETENSORS_WEIGHTS_FILE_PATTERN": "model{suffix}.safetensors",
            "TF2_WEIGHTS_FILE_PATTERN": "tf_model{suffix}.h5",
        }
        for name, value in compat_constants.items():
            if not hasattr(hf_constants, name):
                setattr(hf_constants, name, value)
                log(f"HF_HUB_CONSTANT_PATCH=True {name}={value}")
            else:
                log(f"HF_HUB_CONSTANT_PATCH=False {name}")

        if not hasattr(huggingface_hub, "split_torch_state_dict_into_shards"):
            from huggingface_hub.serialization import split_torch_state_dict_into_shards
            huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards
            log("HF_HUB_TORCH_SHARD_COMPAT_PATCH=True")
        else:
            log("HF_HUB_TORCH_SHARD_COMPAT_PATCH=False")
        log(f"HF_HUB_VERSION={getattr(huggingface_hub, '__version__', 'unknown')}")
    except Exception as exc:
        raise RuntimeError(f"huggingface-hub compatibility patch failed: {type(exc).__name__}: {exc}")
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
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trans_model = trans_model.to(device)
    trans_model.eval()

    # The bundled IndicTrans2 generation code uses the legacy tuple cache.
    # Transformers 4.57 can supply a newer cache structure that contains
    # placeholders, leading to past_key_values[0][0] being None. Disable
    # generation caching for this small segment batch; correctness is more
    # important than cache speed here.
    trans_model.config.use_cache = False
    log("INDICTRANS_USE_CACHE=False")

    source_texts = [s["text"] for s in segments]
    batch = ip.preprocess_batch(source_texts, src_lang=SOURCE_LANG, tgt_lang=TARGET_LANG)
    inputs = tokenizer(
        batch, truncation=True, padding="longest", max_length=256, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = trans_model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=1,
            max_length=256,
            length_penalty=0.6,
            early_stopping=True,
            use_cache=False
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

    (WORK / "r72_segment_manifest.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for i, s in enumerate(segments, 1):
        log(f"TRANSLATION_{i}={s['english']}")

    log("R72 CHECKPOINT 2 PASSED — SEGMENT TRANSLATION + QUALITY BEAM SEARCH")

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
        raw = WORK / f"r72_tts_{i:02d}_raw.mp3"
        fit = WORK / f"r72_tts_{i:02d}_fit.wav"
        awaitable = synthesize(s["english"], VOICE, raw)
        asyncio.run(awaitable)
        actual = media_duration(raw)
        target = max(0.30, s["end"] - s["start"])
        if actual <= 0:
            raise RuntimeError(f"TTS duration unavailable for segment {i}")

        speed = max(1.0, actual / target)
        if speed > MAX_TTS_SPEED:
            log(f"TTS_{i}_QUALITY_WARNING=speed {speed:.3f} exceeds cap {MAX_TTS_SPEED:.2f}")
            speed = MAX_TTS_SPEED
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

    (WORK / "r72_segment_manifest.json").write_text(
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
    dub_audio = WORK / "r72_english_dub_30s.wav"
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", ";".join(filters),
        "-map", "[out]", "-ar", "48000", "-ac", "1",
        "-c:a", "pcm_s16le", str(dub_audio)
    ]
    rc = run_cmd(cmd, timeout=240)
    if rc != 0 or not dub_audio.exists():
        raise RuntimeError("Timed English dub audio build failed")

    log("\n[9] MIX ENGLISH DUB + ORIGINAL MUSIC BED")
    final_audio = WORK / "r72_final_mix_30s.wav"
    rc = run_cmd([
        "ffmpeg", "-y",
        "-i", str(dub_audio), "-i", str(music_bed),
        "-filter_complex",
        "[0:a]volume=1.0[dub];[1:a]volume=0.32[music];"
        "[dub][music]amix=inputs=2:duration=longest:dropout_transition=0,"
        "alimiter=limit=0.95,atrim=duration=30.000,asetpts=PTS-STARTPTS[out]",
        "-map", "[out]", "-ar", "48000", "-ac", "2",
        "-c:a", "pcm_s16le", str(final_audio)
    ], timeout=240)
    if rc != 0 or not final_audio.exists():
        raise RuntimeError("Final English + music mix failed")
    log("MUSIC_BED_MIX_VOLUME=0.32")
    
    log("\n[10] MUX DUBBED VIDEO")
    dubbed_video = WORK / "r72_timed_dub_30s.mp4"
    rc = run_cmd([
        "ffmpeg", "-y", "-ss", "0", "-t", str(CLIP_SECONDS),
        "-i", str(video), "-i", str(final_audio),
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

    log("R72 CHECKPOINT 3 PASSED — INDIAN ENGLISH TTS + TIMED AUDIO BUILD")
    log("R72 CHECKPOINT 3B PASSED — ORIGINAL MUSIC BED RESTORED")
    log("R72 CHECKPOINT 4 PASSED — 30s QUALITY DUBBED VIDEO")
    log("\nR72 DIAGNOSTIC COMPLETE — TIMED TELUGU → ENGLISH DUBBING PASSED")

except Exception as exc:
    log("\n" + "=" * 78)
    log("R72 DIAGNOSTIC FAILED")
    log("=" * 78)
    log(f"{type(exc).__name__}: {exc}")
    log(traceback.format_exc())
    raise
