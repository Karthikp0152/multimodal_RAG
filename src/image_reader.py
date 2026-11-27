import hashlib
import io
import json
import os

from google.genai import types
from PIL import Image
from pydantic import BaseModel

from src.generator import GENERATOR_MODEL, LLM_PROVIDER, get_client, image_part

CACHE_PATH = os.path.join(".cache", "image_text.json")

READ_PROMPT = (
    "This image comes from a page of a document. Classify it and extract its content.\n"
    '- "text": the image mainly shows words (a heading, a name, a title, a scanned '
    "paragraph). Put the exact words in content, preserving their order. Do not add "
    "words that are not visible.\n"
    '- "description": the image shows a figure, chart, diagram, table, or photo. Put a '
    "short factual description of what it shows in content, including any labels or "
    "numbers that are visible.\n"
    '- "decorative": an ornament, divider, logo, or flourish with no information. '
    "Leave content empty."
)


class ImageContent(BaseModel):
    kind: str
    content: str


# Gemini enforces the JSON shape with response_schema; for ASU we ask for it in the prompt.
JSON_INSTRUCTION = (
    '\n\nReply with only a JSON object, no other text: {"kind": "text" | "description" | '
    '"decorative", "content": "..."}'
)


_cache: dict[str, dict] | None = None


def _load_cache() -> dict[str, dict]:
    global _cache
    if _cache is None:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_cache, f, indent=2, ensure_ascii=False)


def to_png(image_bytes: bytes) -> bytes:
    """Convert any format Pillow can open (JPEG, WebP, GIF, BMP, TIFF...) to PNG."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGBA")
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()


def read_image(png_bytes: bytes) -> dict:
    """Return {"kind": "text" | "description" | "decorative", "content": str}.

    Results are cached by image hash, so repeated ornaments and re-runs cost nothing.
    """
    cache = _load_cache()
    key = hashlib.sha256(GENERATOR_MODEL.encode() + READ_PROMPT.encode() + png_bytes).hexdigest()[:20]
    if key in cache:
        return cache[key]

    if LLM_PROVIDER == "asu":
        result = _read_image_asu(png_bytes)
    else:
        result = _read_image_gemini(png_bytes)

    cache[key] = result
    _save_cache()
    return result


def _clean(kind: str | None, content: str | None) -> dict:
    """Keep only the three known kinds; anything else counts as decorative."""
    if kind not in ("text", "description", "decorative"):
        return {"kind": "decorative", "content": ""}
    return {"kind": kind, "content": (content or "").strip()}


def _read_image_gemini(png_bytes: bytes) -> dict:
    response = get_client().models.generate_content(
        model=GENERATOR_MODEL,
        contents=[types.Part.from_bytes(data=png_bytes, mime_type="image/png"), READ_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ImageContent,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        return _clean(None, None)
    return _clean(parsed.kind, parsed.content)


def _read_image_asu(png_bytes: bytes) -> dict:
    response = get_client().chat.completions.create(
        model=GENERATOR_MODEL,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": [image_part(png_bytes), {"type": "text", "text": READ_PROMPT + JSON_INSTRUCTION}],
        }],
    )
    text = response.choices[0].message.content or ""
    start, end = text.find("{"), text.rfind("}")
    try:
        parsed = ImageContent.model_validate_json(text[start:end + 1])
    except Exception:
        return _clean(None, None)
    return _clean(parsed.kind, parsed.content)