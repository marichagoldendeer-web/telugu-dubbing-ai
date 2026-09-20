import os, sys, json, time, traceback, subprocess
from pathlib import Path

OUT=Path("/kaggle/working/automation_result.txt")
WORK=Path("/kaggle/working")
VIDEO_DS="anjiallkumar/r52-telugu-source-video"
TRANS_DS="anjiallkumar/indictrans2-indic-en-dist-200m"
ASR_DS="anjiallkumar/indicconformer-te-hybrid-rnnt-large"

lines=[]
def log(x=""):
    print(x, flush=True); lines.append(str(x))
def save():
    OUT.write_text("\n".join(lines)+"\n", encoding="utf-8")

def candidates(slug):
    owner, name=slug.split("/",1)
    return [
        Path("/kaggle/input")/name,
        Path("/kaggle/input")/slug,
    ]

def locate(slug, marker=None):
    for p in candidates(slug):
        if p.exists():
            return p
    # Kaggle API-pushed kernels may expose attached datasets below
    # /kaggle/input/datasets instead of /kaggle/input/<slug>.
    base=Path("/kaggle/input")
    if not base.exists():
        return None
    owner, name=slug.split("/",1)
    direct_matches=[p for p in base.rglob("*") if p.is_dir() and p.name==name]
    if direct_matches:
        return direct_matches[0]
    if marker:
        matches=[]
        for p in base.rglob(marker):
            if p.is_file():
                matches.append(p.parent)
        if matches:
            return matches[0]
    return None

def download_dataset(slug, target):
    log(f"Attempting runtime recovery with kagglehub: {slug}")
    try:
        import kagglehub
    except Exception:
        subprocess.run([sys.executable,"-m","pip","install","-q","kagglehub"],check=True)
        import kagglehub
    target.mkdir(parents=True, exist_ok=True)
    p=kagglehub.dataset_download(slug, output_dir=str(target), force_download=False)
    log(f"kagglehub returned: {p}")
    return Path(p)

try:
    log("="*70)
    log("R53 — DATASET MOUNT DIAGNOSTIC + INDICTRANS2 RECOVERY")
    log("="*70)

    log("\n[1] Runtime")
    log(sys.version.splitlines()[0])

    log("\n[2] /kaggle/input top-level entries")
    inp=Path("/kaggle/input")
    if inp.exists():
        for p in sorted(inp.iterdir()):
            log(f"{p}  {'DIR' if p.is_dir() else 'FILE'}")
    else:
        raise FileNotFoundError("/kaggle/input does not exist")

    log("\n[3] Dataset mount check")
    roots={}
    for label,slug in [("VIDEO",VIDEO_DS),("INDICTRANS",TRANS_DS),("ASR",ASR_DS)]:
        p=locate(slug)
        roots[label]=p
        log(f"{label}: {p if p else 'MISSING'}")

    if roots["VIDEO"] is None:
        raise FileNotFoundError("Video dataset is not mounted anywhere under /kaggle/input")
    if roots["ASR"] is None:
        raise FileNotFoundError("IndicConformer dataset is not mounted anywhere under /kaggle/input")

    if roots["INDICTRANS"] is None:
        log("\n[4] IndicTrans2 attachment missing — starting runtime recovery")
        recovery=WORK/"r53_recovered"
        roots["INDICTRANS"]=download_dataset(TRANS_DS,recovery)
    else:
        log("\n[4] IndicTrans2 attachment present; no recovery needed")

    trans=roots["INDICTRANS"]
    if not trans.exists():
        raise FileNotFoundError(f"IndicTrans2 recovery failed: {trans}")

    log(f"IndicTrans2 root: {trans}")
    files=[p for p in trans.rglob("*") if p.is_file()]
    log(f"IndicTrans2 files found: {len(files)}")
    for p in files[:30]:
        log(f"  {p.relative_to(trans)}")

    model_markers=[p for p in files if p.name in {
        "config.json","tokenizer_config.json","spiece.model",
        "tokenizer.json","model.safetensors","pytorch_model.bin"
    }]
    if not model_markers:
        raise FileNotFoundError("IndicTrans2 root has no recognizable Hugging Face model/tokenizer files")

    log("\n[5] Video and ASR contents")
    vids=list(roots["VIDEO"].rglob("*.mp4"))
    nemo=list(roots["ASR"].rglob("*.nemo"))
    log(f"MP4 files: {len(vids)}")
    log(f"Nemo checkpoints: {len(nemo)}")
    if vids: log(f"Video: {vids[0]}")
    if nemo: log(f"ASR checkpoint: {nemo[0]}")
    if not vids: raise FileNotFoundError("No MP4 found")
    if not nemo: raise FileNotFoundError("No .nemo checkpoint found")

    log("\n[6] GPU")
    gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],capture_output=True,text=True)
    log(gpu.stdout.strip())
    if gpu.returncode!=0: raise RuntimeError("NVIDIA GPU not detected")

    log("\n[7] Checkpoint")
    log("R53 DATASET MOUNT CHECK PASSED")
    log(f"IndicTrans2 source mode: {'ATTACHED' if locate(TRANS_DS) else 'RUNTIME_RECOVERY'}")
    log(f"Video root: {roots['VIDEO']}")
    log(f"ASR root: {roots['ASR']}")
    log(f"IndicTrans2 root: {trans}")
    log("="*70)
    log("R53 CHECKPOINT PASSED — DATASETS AVAILABLE")
    log("="*70)
    save()

except Exception as exc:
    log("\n"+"="*70)
    log("R53 CHECKPOINT FAILED")
    log("="*70)
    log(f"{type(exc).__name__}: {exc}")
    tb=traceback.format_exc()
    print(tb,flush=True)
    lines.extend(tb.rstrip().splitlines())
    save()
    raise
