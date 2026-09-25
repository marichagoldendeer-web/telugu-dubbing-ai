import json
import re
import subprocess
import sys
from pathlib import Path

R81_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("r83-r81-output")
SOURCE_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("r83-source")
REPORT = Path("r83-report")
LISTENING = Path("r83-listening")
REPORT.mkdir(parents=True, exist_ok=True)
LISTENING.mkdir(parents=True, exist_ok=True)


def run(cmd, check=False):
    p = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stdout}"
        )
    return p


def newest(root, pattern):
    items = sorted(
        root.rglob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return items[0] if items else None


def probe_duration(path):
    p = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ], check=True)
    return float(p.stdout.strip())


def probe_streams(path):
    p = run([
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,sample_rate,channels",
        "-of", "json",
        str(path),
    ], check=True)
    return json.loads(p.stdout)


dubbed = newest(R81_ROOT, "*.mp4")
source_candidates = sorted(
    [
        p for p in SOURCE_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in {".webm", ".mp4", ".mkv", ".mov"}
    ]
)
source = source_candidates[0] if source_candidates else None
manifest = newest(R81_ROOT, "r81_segment_manifest.json")
final_qc = newest(R81_ROOT, "r81_final_qc_report.json")
final_mix = newest(R81_ROOT, "r81_final_mix_native.wav")

if not dubbed:
    raise RuntimeError("R83 could not find R81 final dubbed MP4")
if not source:
    raise RuntimeError("R83 could not find the exact R81 source video")
if not manifest:
    raise RuntimeError("R83 could not find r81_segment_manifest.json")
if not final_qc:
    raise RuntimeError("R83 could not find r81_final_qc_report.json")

segments = json.loads(manifest.read_text(encoding="utf-8"))
qc = json.loads(final_qc.read_text(encoding="utf-8"))

source_duration = probe_duration(source)
dubbed_duration = probe_duration(dubbed)

# Use the exact source timeline and choose three representative windows.
# Beginning: 0-20s, middle: centered on 50%, ending: last 20s.
points = [
    ("BEGINNING", 0.0),
    ("MIDDLE", max(0.0, source_duration / 2.0 - 10.0)),
    ("ENDING", max(0.0, source_duration - 20.0)),
]

created = []
for label, start in points:
    original_out = LISTENING / f"r83_{label.lower()}_original_20s.mp4"
    dubbed_out = LISTENING / f"r83_{label.lower()}_dubbed_20s.mp4"

    for inp, out in [(source, original_out), (dubbed, dubbed_out)]:
        p = run([
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", "20",
            "-i", str(inp),
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(out),
        ])
        if p.returncode != 0 or not out.exists():
            raise RuntimeError(f"Failed to create {label} listening sample: {out}")
        created.append(out)

    # Also export isolated WAVs for clean audio-only listening.
    original_wav = LISTENING / f"r83_{label.lower()}_original_20s.wav"
    dubbed_wav = LISTENING / f"r83_{label.lower()}_dubbed_20s.wav"
    for inp, out in [(source, original_wav), (final_mix, dubbed_wav)]:
        if inp is None or not inp.exists():
            continue
        p = run([
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", "20",
            "-i", str(inp),
            "-ar", "48000",
            "-ac", "2",
            "-c:a", "pcm_s16le",
            str(out),
        ])
        if p.returncode != 0:
            raise RuntimeError(f"Failed to create audio sample: {out}")

# Generate a compact machine-readable listening index.
index = {
    "audit": "R83",
    "purpose": "Human listening QC package for the successful R81 genuine longform dub",
    "source_video": str(source),
    "dubbed_video": str(dubbed),
    "source_duration_seconds": source_duration,
    "dubbed_duration_seconds": dubbed_duration,
    "duration_delta_seconds": abs(source_duration - dubbed_duration),
    "r81_final_qc": qc,
    "segments": len(segments),
    "sample_windows": [
        {
            "label": label,
            "start_seconds": start,
            "duration_seconds": 20.0,
            "original_video": f"r83_{label.lower()}_original_20s.mp4",
            "dubbed_video": f"r83_{label.lower()}_dubbed_20s.mp4",
            "original_audio": f"r83_{label.lower()}_original_20s.wav",
            "dubbed_audio": f"r83_{label.lower()}_dubbed_20s.wav",
        }
        for label, start in points
    ],
}

(REPORT / "r83_listening_index.json").write_text(
    json.dumps(index, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

readme = f"""R83 HUMAN LISTENING QC
=======================

Purpose
-------
This package is the human-listening stage for the successful R81 native-source
Telugu -> English longform dub. R79/R80/R81 production files are not modified.

Exact source duration : {source_duration:.3f}s
Dubbed video duration : {dubbed_duration:.3f}s
Duration delta        : {abs(source_duration - dubbed_duration):.3f}s
Segments              : {len(segments)}

Three 20-second A/B windows
---------------------------
1. BEGINNING: 0.000s
2. MIDDLE:    {source_duration / 2.0 - 10.0:.3f}s
3. ENDING:    {max(0.0, source_duration - 20.0):.3f}s

For each window, listen to the ORIGINAL file first and the DUBBED file second.

Human checks
------------
A. Translation naturalness
   Does the English sound like natural spoken English for the context?

B. Pronunciation / names
   Listen for Telugu names, places, numbers, and technical terms.

C. Timing / sync
   Speech should enter and finish close to the visible speaker/action timing.

D. Voice consistency
   Check that the same English voice remains stable across the three windows.

E. Speech / music balance
   English speech should remain clearly intelligible without the background
   music/ambience becoming distracting.

F. Artifacts
   Note clipping, sudden volume jumps, robotic fragments, repeated words,
   missing words, or unnatural pauses.

Recommended human review record
-------------------------------
BEGINNING: Pass / Issue
MIDDLE:    Pass / Issue
ENDING:    Pass / Issue

For each issue, record:
- window
- approximate timestamp
- what you hear/see
- severity: minor / moderate / major
"""

(REPORT / "R83_HUMAN_REVIEW.txt").write_text(readme, encoding="utf-8")

summary = [
    "R83 HUMAN LISTENING QC PACKAGE",
    "==============================",
    "Status: READY_FOR_HUMAN_REVIEW",
    f"Exact source duration: {source_duration:.3f}s",
    f"Dubbed duration: {dubbed_duration:.3f}s",
    f"Duration delta: {abs(source_duration - dubbed_duration):.3f}s",
    f"Segment count: {len(segments)}",
    "Windows: beginning / middle / ending, 20s each",
    f"Files generated: {len(created)}",
    "",
    "Human review is required for translation naturalness, pronunciation, sync,",
    "voice consistency, and speech/music balance. Automated QC does not replace it.",
]
(REPORT / "r83_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

print("\n".join(summary))
