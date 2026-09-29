from pathlib import Path

from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    data_dir: Path = ROOT / "data"
    vendor_dir: Path = ROOT / "vendor"

    asr_provider: str = "local"  # local | openai
    translate_provider: str = "local"  # local | openai
    tts_provider: str = "local"  # local | elevenlabs | sarvam
    lipsync_provider: str = "local"  # local | sync | none

    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    nllb_model: str = "facebook/nllb-200-distilled-600M"

    wav2lip_checkpoint: str = "wav2lip_gan.pth"
    wav2lip_resize_factor: int = 1
    wav2lip_batch_size: int = 16
    # Downscale long edge before lip sync to keep CPU inference tractable.
    lipsync_max_width: int = 720

    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    sarvam_api_key: str = ""
    sync_api_key: str = ""

    class Config:
        env_file = ".env"
        env_prefix = ""


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)

TARGET_LANGUAGES = {
    "te": {"name": "Telugu", "nllb": "tel_Telu", "mms": "facebook/mms-tts-tel", "sarvam": "te-IN"},
    "ta": {"name": "Tamil", "nllb": "tam_Taml", "mms": "facebook/mms-tts-tam", "sarvam": "ta-IN"},
    "kn": {"name": "Kannada", "nllb": "kan_Knda", "mms": "facebook/mms-tts-kan", "sarvam": "kn-IN"},
    "ml": {"name": "Malayalam", "nllb": "mal_Mlym", "mms": "facebook/mms-tts-mal", "sarvam": "ml-IN"},
}

# Whisper ISO-639-1 -> NLLB (FLORES-200) codes for the source side.
NLLB_SOURCE_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "id": "ind_Latn",
    "tr": "tur_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
}

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi",
    "ur": "Urdu", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "it": "Italian", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic", "id": "Indonesian", "tr": "Turkish", "nl": "Dutch", "pl": "Polish",
    "vi": "Vietnamese", "th": "Thai",
}
