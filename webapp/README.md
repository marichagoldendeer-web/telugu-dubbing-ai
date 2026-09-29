# Dubbing Studio

Upload a video, detect the spoken language, and get it back dubbed into **Telugu, Tamil, Kannada or Malayalam** with the speaker's mouth re-synced to the new audio.

Everything runs locally on CPU by default (no API keys required); each stage can be swapped for a hosted provider through environment variables.

## Pipeline

| Stage | Default (local, free) | Optional hosted provider |
| --- | --- | --- |
| Language detection + transcription | faster-whisper (`small`, int8) | OpenAI `whisper-1` |
| Translation | NLLB-200 distilled 600M | OpenAI `gpt-4o-mini` |
| Text to speech | Meta MMS-TTS (`mms-tts-tel/tam/kan/mal`) | ElevenLabs, Sarvam AI |
| Lip sync | Wav2Lip (GAN checkpoint) | sync.so |

Each transcript segment is synthesized separately and time-stretched (pitch preserving, 0.7x–1.6x)
so the dubbed line lands in the same slot as the original line, then all clips are mixed onto a
silent canvas the length of the video. The mixed track drives Wav2Lip, which re-renders the mouth
region frame by frame.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/setup_wav2lip.sh          # clones Wav2Lip + downloads weights (~500 MB)
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000, drop in a video, pick a language, hit **Start dubbing**.
ASR/translation/TTS weights (~4 GB total) download from Hugging Face on first use.

## Speed

On this 8-core CPU box a 17 s clip takes ~6 minutes end to end, of which ~4 minutes is Wav2Lip.
With a GPU, Wav2Lip is roughly 20-40x faster. Knobs in `app/config.py` (all overridable via env):

- `LIPSYNC_MAX_WIDTH` (default 720) — video is downscaled before lip sync
- `WAV2LIP_BATCH_SIZE`, `WHISPER_MODEL` (`tiny`/`base`/`small`/`medium`)

## Configuration

Copy `.env.example` to `.env`:

```
ASR_PROVIDER=local|openai
TRANSLATE_PROVIDER=local|openai
TTS_PROVIDER=local|elevenlabs|sarvam
LIPSYNC_PROVIDER=local|sync|none
OPENAI_API_KEY=...
ELEVENLABS_API_KEY=...
SARVAM_API_KEY=...
SYNC_API_KEY=...
```

## API

- `POST /api/jobs` — multipart: `file`, `target_language` (`te|ta|kn|ml`), `lip_sync` (bool)
- `GET /api/jobs/{id}` — status, progress, transcript + translations
- `GET /api/jobs/{id}/video` — dubbed mp4

## Limitations

- One speaker at a time: there is no diarization, so a single TTS voice covers everyone.
- Wav2Lip needs a visible, detectable face; if none is found the job still returns the dubbed
  audio muxed over the original video and reports a warning.
- MMS-TTS voices are robotic compared with ElevenLabs/Sarvam — switch `TTS_PROVIDER` if quality matters.
- Jobs live in memory; restarting the server forgets them (files remain under `data/jobs/`).
