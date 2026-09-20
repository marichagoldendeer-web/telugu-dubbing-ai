import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path("/kaggle/working/automation_result.txt")
TARGET = Path("/tmp/indictrans_pkgs")
HF_CACHE = Path("/tmp/huggingface_cache")

R49_REQUIREMENTS = """huggingface-hub==0.36.2
transformers==4.57.6
indictranstoolkit==1.1.1
"""

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
SRC_LANG = "tel_Telu"
TGT_LANG = "eng_Latn"

INPUT_SENTENCES = [
    "నేను ఈ రోజు మార్కెట్‌కు వెళ్తున్నాను.",
    "మన తెలుగు డబ్బింగ్ ప్రాజెక్ట్ ఇప్పుడు గిట్‌హబ్ నుంచి కాగుల్ జీపీయూ మీద స్వయంచాలకంగా నడుస్తోంది.",
]

lines = []

def log(message=""):
    print(message, flush=True)
    lines.append(str(message))

def save():
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ Result saved to {OUT}", flush=True)

def run_checked(cmd, label):
    log(f"\n{label}")
    log("$ " + " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
        lines.extend(result.stdout.rstrip().splitlines())
    if result.stderr:
        print(result.stderr.rstrip(), flush=True)
        lines.extend(result.stderr.rstrip().splitlines())
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")

try:
    log("=" * 70)
    log("R50 — INDIC TRANS2 TELUGU → ENGLISH GPU TRANSLATION TEST")
    log("=" * 70)

    log("\n[1] Runtime")
    log(sys.version.splitlines()[0])
    log(platform.platform())

    log("\n[2] GPU")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if gpu.returncode != 0:
        raise RuntimeError("NVIDIA GPU not detected")
    log(gpu.stdout.strip())

    log("\n[3] PyTorch / CUDA")
    import torch

    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"CUDA version: {torch.version.cuda}")
    log(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    log("\n[4] R49 isolated dependencies")
    TARGET.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    req_file = Path("/tmp/r49-requirements.txt")
    req_file.write_text(R49_REQUIREMENTS, encoding="utf-8")
    log(R49_REQUIREMENTS.strip())

    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            "--target",
            str(TARGET),
            "-r",
            str(req_file),
        ],
        "Installing R49 packages",
    )

    sys.path.insert(0, str(TARGET))
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")

    log("\n[5] Dependency versions")
    import huggingface_hub
    import transformers
    from IndicTransToolkit import IndicProcessor

    log(f"huggingface_hub: {huggingface_hub.__version__}")
    log(f"transformers: {transformers.__version__}")
    log("IndicTransToolkit: import OK")

    log("\n[6] Load IndicTrans2 model")
    log(f"Model: {MODEL_NAME}")
    log(f"Source: {SRC_LANG}")
    log(f"Target: {TGT_LANG}")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda"
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(HF_CACHE),
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir=str(HF_CACHE),
    ).to(device)

    model.eval()

    load_seconds = time.time() - load_start
    log(f"Model loaded in {load_seconds:.2f} seconds")
    log(f"Model device: {next(model.parameters()).device}")
    log(f"Model dtype: {next(model.parameters()).dtype}")

    log("\n[7] Telugu → English translation")
    ip = IndicProcessor(inference=True)
    batch = ip.preprocess_batch(
        INPUT_SENTENCES,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
    )

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    infer_start = time.time()
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )
    infer_seconds = time.time() - infer_start

    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    translations = ip.postprocess_batch(decoded, lang=TGT_LANG)

    log(f"Inference time: {infer_seconds:.2f} seconds")
    log(f"Translated items: {len(translations)}")

    results = []
    for source, translation in zip(INPUT_SENTENCES, translations):
        log("")
        log(f"TELUGU: {source}")
        log(f"ENGLISH: {translation}")
        results.append({"source": source, "translation": translation})

    log("\n[8] Kaggle environment")
    log("KAGGLE_KERNEL_RUN_TYPE: " + str(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")))
    log("KAGGLE_URL_BASE: " + str(os.environ.get("KAGGLE_URL_BASE")))

    log("\n[9] Result JSON")
    result_json = {
        "model": MODEL_NAME,
        "source_language": SRC_LANG,
        "target_language": TGT_LANG,
        "device": str(next(model.parameters()).device),
        "model_dtype": str(next(model.parameters()).dtype),
        "model_load_seconds": round(load_seconds, 2),
        "inference_seconds": round(infer_seconds, 2),
        "translations": results,
    }
    log(json.dumps(result_json, ensure_ascii=False, indent=2))

    log("\n" + "=" * 70)
    log("R50 CHECKPOINT PASSED — REAL TELUGU → ENGLISH MODEL INFERENCE")
    log("=" * 70)
    save()

except Exception as exc:
    log("\n" + "=" * 70)
    log("R50 CHECKPOINT FAILED")
    log("=" * 70)
    log(f"{type(exc).__name__}: {exc}")
    log("\nTraceback:")
    tb = traceback.format_exc()
    print(tb, flush=True)
    lines.extend(tb.rstrip().splitlines())
    save()
    raise
