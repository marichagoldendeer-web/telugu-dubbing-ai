#!/usr/bin/env bash
# Fetch the Wav2Lip inference code and model weights into vendor/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor"
W2L="$VENDOR/Wav2Lip"
HF="https://huggingface.co/camenduru/Wav2Lip/resolve/main"

mkdir -p "$VENDOR"
if [ ! -d "$W2L/.git" ]; then
  git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git "$W2L"
fi

mkdir -p "$W2L/checkpoints"
if [ ! -s "$W2L/checkpoints/wav2lip_gan.pth" ]; then
  curl -L --fail -o "$W2L/checkpoints/wav2lip_gan.pth" "$HF/checkpoints/wav2lip_gan.pth"
fi

S3FD="$W2L/face_detection/detection/sfd/s3fd.pth"
if [ ! -s "$S3FD" ]; then
  curl -L --fail -o "$S3FD" "$HF/face_detection/detection/sfd/s3fd.pth"
fi

# Wav2Lip targets librosa 0.7 / numpy 1.17 APIs; patch for the pinned versions.
python3 - "$W2L" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
audio = root / "audio.py"
text = audio.read_text()
text = text.replace(
    "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,",
    "return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,",
)
text = text.replace(
    "return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)",
    "return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)",
)
audio.write_text(text)

for path in root.rglob("*.py"):
    body = path.read_text()
    patched = body.replace("np.float)", "float)").replace("np.int)", "int)")
    if patched != body:
        path.write_text(patched)
PY

echo "Wav2Lip ready at $W2L"
