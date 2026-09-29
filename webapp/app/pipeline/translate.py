from functools import lru_cache

from ..config import LANGUAGE_NAMES, NLLB_SOURCE_CODES, TARGET_LANGUAGES, settings

_NAME_TO_CODE = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}


def normalize_language(language: str) -> str:
    """faster-whisper returns ISO codes, the OpenAI API returns names like 'english'."""
    language = (language or "").strip().lower()
    if language in LANGUAGE_NAMES:
        return language
    return _NAME_TO_CODE.get(language, language[:2])


@lru_cache(maxsize=1)
def _nllb():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.nllb_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.nllb_model)
    model.eval()
    return tokenizer, model


def _translate_local(texts: list[str], source: str, target: str) -> list[str]:
    import torch

    tokenizer, model = _nllb()
    tokenizer.src_lang = NLLB_SOURCE_CODES.get(source, "eng_Latn")
    target_code = TARGET_LANGUAGES[target]["nllb"]
    bos = tokenizer.convert_tokens_to_ids(target_code)

    out: list[str] = []
    for start in range(0, len(texts), 8):
        batch = texts[start : start + 8]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.inference_mode():
            generated = model.generate(
                **encoded, forced_bos_token_id=bos, max_new_tokens=256, num_beams=2
            )
        out.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return out


def _translate_openai(texts: list[str], source: str, target: str) -> list[str]:
    import json

    import httpx

    target_name = TARGET_LANGUAGES[target]["name"]
    source_name = LANGUAGE_NAMES.get(source, source)
    prompt = (
        f"Translate each numbered {source_name} subtitle line into natural spoken {target_name}. "
        "Keep each translation roughly the same spoken length as the source so it fits the same "
        "time slot. Reply with JSON: {\"lines\": [{\"id\": 1, \"text\": \"...\"}]}.\n\n"
        + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    )
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        },
        timeout=300,
    )
    response.raise_for_status()
    lines = json.loads(response.json()["choices"][0]["message"]["content"])["lines"]
    by_id = {int(line["id"]): line["text"] for line in lines}
    return [by_id.get(i + 1, "") for i in range(len(texts))]


def translate(texts: list[str], source: str, target: str) -> list[str]:
    if not texts:
        return []
    if settings.translate_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return _translate_openai(texts, source, target)
    return _translate_local(texts, source, target)
