import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("r82-output")
REPORT = Path("r82-report")
SAMPLES = REPORT / "samples"
REPORT.mkdir(parents=True, exist_ok=True)
SAMPLES.mkdir(parents=True, exist_ok=True)

def run(cmd, check=False):
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, check=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stdout}")
    return p

def newest(pattern):
    items = sorted(ROOT.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return items[0] if items else None

video = newest("*.mp4")
audio = newest("r81_final_mix_native.wav") or newest("*.wav")
manifest = newest("r81_segment_manifest.json")
audit = newest("r81_quality_audit_report.json")
final_qc = newest("r81_final_qc_report.json")

if not video:
    raise RuntimeError("R82 could not find the R81 final MP4")
if not manifest:
    raise RuntimeError("R82 could not find r81_segment_manifest.json")

segments = json.loads(manifest.read_text(encoding="utf-8"))
prior_audit = json.loads(audit.read_text(encoding="utf-8")) if audit else {}
prior_final_qc = json.loads(final_qc.read_text(encoding="utf-8")) if final_qc else {}

probe = run([
    "ffprobe", "-v", "error", "-show_entries",
    "format=duration,size,format_name:stream=index,codec_type,codec_name,sample_rate,channels,width,height",
    "-of", "json", str(video)
], check=True)
info = json.loads(probe.stdout)
duration = float(info.get("format", {}).get("duration", 0.0))
size = int(info.get("format", {}).get("size", 0))
streams = info.get("streams", [])
vstreams = [s for s in streams if s.get("codec_type") == "video"]
astreams = [s for s in streams if s.get("codec_type") == "audio"]

# Prefer the source duration already verified by R81's final QC report.
# The manifest's final source_end is the end of the last spoken segment, not
# necessarily the end of the media container. R81's verified duration therefore
# avoids falsely flagging legitimate lead-in / tail regions as duration drift.
manifest_source_end = max(
    (float(s.get("source_end", s.get("end", 0))) for s in segments),
    default=0.0,
)
if float(prior_final_qc.get("duration_seconds", 0.0) or 0.0) > 0:
    source_duration = float(prior_final_qc["duration_seconds"])
    source_duration_basis = "r81_final_qc_report.duration_seconds"
else:
    source_duration = manifest_source_end
    source_duration_basis = "manifest.final_source_end_fallback"

duration_delta = abs(duration - source_duration) if source_duration else None
manifest_tail_gap = source_duration - manifest_source_end if source_duration else None

speed_values = [float(s.get("speed_factor", 1.0)) for s in segments if s.get("speed_factor") is not None]
wps_values = []
for s in segments:
    target = float(s.get("target_duration") or max(0.01, float(s.get("end", 0)) - float(s.get("start", 0))))
    words = re.findall(r"[A-Za-z0-9']+", str(s.get("english", "")))
    if target > 0:
        wps_values.append(len(words) / target)

warnings = []
errors = []

if len(vstreams) != 1:
    errors.append(f"expected 1 video stream, found {len(vstreams)}")
if len(astreams) != 1:
    errors.append(f"expected 1 audio stream, found {len(astreams)}")
if size < 100_000:
    errors.append(f"final MP4 is unexpectedly small: {size} bytes")
if duration_delta is not None and duration_delta > 0.10:
    errors.append(f"duration drift is {duration_delta:.3f}s")
if max(speed_values, default=1.0) > 1.30 + 1e-6:
    errors.append(f"observed TTS speed factor exceeds 1.30x: {max(speed_values):.3f}x")
if max(wps_values, default=0.0) > 5.5:
    warnings.append(f"at least one segment exceeds 5.5 English words/s: {max(wps_values):.2f}")
if len(segments) < 20:
    warnings.append(f"unusually low segment count: {len(segments)}")
if manifest_tail_gap is not None and manifest_tail_gap > 0.10:
    warnings.append(
        f"manifest speech coverage ends {manifest_tail_gap:.3f}s before verified media duration; "
        "treated as source tail rather than duration drift"
    )

# Detect long digital-silence regions in the final audio stream.
silence = None
if audio and audio.exists():
    silence = run([
        "ffmpeg", "-hide_banner", "-i", str(audio),
        "-af", "silencedetect=noise=-45dB:d=1.50",
        "-f", "null", "-"
    ])
    silence_count = len(re.findall(r"silence_start:", silence.stdout))
    if silence.returncode != 0:
        warnings.append("ffmpeg silencedetect did not complete cleanly")
else:
    silence_count = None
    warnings.append("R81 final mix WAV was not present; silence audit skipped")

# Loudness audit of the final mix. This is diagnostic, not a perceptual verdict.
loudness = None
if audio and audio.exists():
    loud = run([
        "ffmpeg", "-hide_banner", "-i", str(audio),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-"
    ])
    match = re.search(r'\{\s*"input_i".*?\}', loud.stdout, flags=re.S)
    if match:
        try:
            loudness = json.loads(match.group(0))
        except Exception:
            pass
    if loudness is None:
        warnings.append("loudnorm diagnostic did not return parseable JSON")

# Extract three representative 20s listening samples: beginning, middle, end.
sample_points = sorted(set([
    0.0,
    max(0.0, duration / 2.0 - 10.0),
    max(0.0, duration - 20.0),
]))
if audio and audio.exists():
    for i, start in enumerate(sample_points, 1):
        out = SAMPLES / f"r82_sample_{i:02d}_{start:.1f}s.wav"
        p = run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", "20",
            "-i", str(audio), "-ar", "48000", "-ac", "2",
            "-c:a", "pcm_s16le", str(out)
        ])
        if p.returncode != 0:
            warnings.append(f"could not create listening sample {i}")

report = {
    "audit": "R82",
    "status": "PASS" if not errors else "FAIL",
    "purpose": "Post-production audit of the successful R81 genuine longform output",
    "input_video": str(video),
    "input_audio": str(audio) if audio else None,
    "duration_seconds": duration,
    "verified_source_duration_seconds": source_duration,
    "source_duration_basis": source_duration_basis,
    "manifest_final_source_end_seconds": manifest_source_end,
    "manifest_tail_gap_seconds": manifest_tail_gap,
    "duration_delta_seconds": duration_delta,
    "file_size_bytes": size,
    "video_stream_count": len(vstreams),
    "audio_stream_count": len(astreams),
    "segment_count": len(segments),
    "max_tts_speed_factor": max(speed_values, default=1.0),
    "average_tts_speed_factor": sum(speed_values) / len(speed_values) if speed_values else None,
    "max_english_words_per_second": max(wps_values, default=0.0),
    "long_silence_events": silence_count,
    "loudness_measurement": loudness,
    "r81_quality_report": prior_audit,
    "r81_final_qc_report": prior_final_qc,
    "warnings": warnings,
    "errors": errors,
}

(REPORT / "r82_quality_audit_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)

summary = [
    "R82 QUALITY AUDIT",
    "==================",
    f"Status: {report['status']}",
    f"Video: {video}",
    f"Duration: {duration:.3f}s",
    f"Verified source duration: {source_duration:.3f}s",
    f"Source duration basis: {source_duration_basis}",
    f"Manifest final speech boundary: {manifest_source_end:.3f}s",
    f"Manifest tail gap: {manifest_tail_gap:.3f}s" if manifest_tail_gap is not None else "Manifest tail gap: unavailable",
    f"Duration delta: {duration_delta:.3f}s" if duration_delta is not None else "Duration delta: unavailable",
    f"Segments: {len(segments)}",
    f"Max TTS speed: {max(speed_values, default=1.0):.3f}x",
    f"Max English density: {max(wps_values, default=0.0):.3f} words/s",
    f"Long silence events: {silence_count if silence_count is not None else 'skipped'}",
    "",
    "Warnings:",
    *([f"- {x}" for x in warnings] or ["- none"]),
    "",
    "Errors:",
    *([f"- {x}" for x in errors] or ["- none"]),
    "",
    "NOTE: Automated metrics verify technical integrity and timing pressure.",
    "They do not replace human listening for translation naturalness, pronunciation,",
    "voice identity consistency, or subjective speech/music balance.",
]
(REPORT / "r82_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

print("\n".join(summary))
if errors:
    sys.exit(1)
