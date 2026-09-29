# Devin integration notes

This branch keeps the existing GPU/quality-oriented dubbing experiments intact and
adopts only the software-engineering pieces that are useful to the production pipeline.

## Imported from the Devin prototype

- Explicit input handling instead of selecting an arbitrary/latest MP4.
- SHA-256 input fingerprinting so every run can be traced to the exact source file.
- A unique directory for every run, preventing stale intermediate-file reuse.
- A persistent JSON run manifest with stage status, timestamps, errors, and outputs.
- Resume support: a later process can reopen a run and inspect which stages passed.
- Per-stage output registration and validation.
- Environment-variable provider knobs so future ASR/translation/TTS/lip-sync providers can be swapped without changing the run-state layer.
- Segment timing math for pitch-preserving FFmpeg `atempo` correction.

## Deliberately NOT imported

The Devin PR's CPU-oriented model stack is not being merged into the Kaggle GPU environment:

- faster-whisper `small/int8` as the default ASR
- NLLB-200 distilled 600M as the translation engine
- MMS-TTS as the production voice
- Devin's pinned Torch/NumPy/librosa stack
- Wav2Lip setup script and its legacy dependency pins

Those choices belong to a separate prototype and would risk recreating the
dependency conflicts already encountered in the GPU project.

## Why this matters for our project

The existing R84-era work remains the source of truth for GPU/model experiments.
The new runtime layer sits around it:

```
exact input
   ↓
SHA-256 fingerprint
   ↓
isolated run directory
   ↓
stage manifest
   ↓
existing GPU dubbing stages
   ↓
per-stage outputs/QC
   ↓
resume after failure/restart
```

This gives us the reliability of the Devin application's orchestration without
throwing away the model work already done in Kaggle.
