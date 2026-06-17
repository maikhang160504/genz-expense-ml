import json
import os
import time
import urllib.request

import google.genai as genai

try:
    from google.genai import errors as genai_errors
except ImportError:  # pragma: no cover
    genai_errors = None

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None


def ensure_gemini_system_instruction(payload: dict, system_text: str) -> None:
    sys_obj = payload.get("systemInstruction")
    if isinstance(sys_obj, str):
        return
    if isinstance(sys_obj, dict) and sys_obj.get("parts"):
        return
    payload["systemInstruction"] = system_text


def _gemini_fallback_models(primary: str) -> list[str]:
    env = os.environ.get("GEMINI_MODEL_FALLBACK", "")
    extra = [m.strip() for m in env.split(",") if m.strip()]
    defaults = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for m in [primary.replace("models/", ""), *extra, *defaults]:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _is_quota_or_rate_limit(exc: BaseException) -> bool:
    """429 / hết quota — đổi API key trong .env, không retry cùng key."""
    if genai_errors and isinstance(
        exc, (genai_errors.ClientError, genai_errors.ServerError)
    ):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code == 429:
            return True
    msg = str(exc).lower()
    return any(
        x in msg
        for x in ("429", "resource exhausted", "resource_exhausted", "quota", "rate limit")
    )


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    if _is_quota_or_rate_limit(exc):
        return False
    msg = str(exc).lower()
    if "high demand" in msg:
        return False
    if genai_errors and isinstance(exc, genai_errors.ServerError):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in (500, 503, 504):
            return True
    return any(x in msg for x in ("503", "unavailable", "504"))


def _call_gemini_once(client, model_name: str, contents, config) -> dict:
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return json.loads(json.dumps(response, default=str))


def call_gemini(api_key: str, model: str, payload: dict) -> dict:
    """Gọi Gemini; retry 503/429 và thử model dự phòng."""
    client = genai.Client(api_key=api_key)
    generation_config = dict(payload.get("generationConfig") or {})
    system_instruction = payload.get("systemInstruction")
    if isinstance(system_instruction, dict) and system_instruction.get("parts"):
        parts = system_instruction.get("parts") or []
        system_instruction = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    contents = payload.get("contents")
    if system_instruction:
        generation_config.setdefault("systemInstruction", system_instruction)
    config = genai.types.GenerateContentConfig(**generation_config)

    max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", "1"))
    base_sleep = float(os.environ.get("GEMINI_RETRY_SLEEP", "3"))
    models = _gemini_fallback_models(model)
    last_exc: BaseException | None = None

    for model_name in models:
        for attempt in range(max_retries):
            try:
                return _call_gemini_once(client, model_name, contents, config)
            except Exception as exc:
                last_exc = exc
                if _is_quota_or_rate_limit(exc):
                    raise
                if not _is_retryable_gemini_error(exc):
                    raise
                wait = base_sleep * (2**attempt)
                print(
                    f"[gemini] {model_name} attempt {attempt + 1}/{max_retries}: "
                    f"{exc!s:.120} — retry in {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
        print(f"[gemini] model {model_name} failed, trying fallback…", flush=True)

    if last_exc:
        raise last_exc
    raise RuntimeError("Gemini call failed with no exception")


def _call_groq_http(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> dict:
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_lmstudio(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.15,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> dict:
    """OpenAI-compatible chat API (LM Studio, Ollama, v.v.)."""
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/v1/chat/completions" if "/v1" not in url else f"{url}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_chat_text(resp: dict) -> str:
    """Lấy text từ response OpenAI hoặc Gemini."""
    if not resp:
        return ""
    choices = resp.get("choices") or []
    for c in choices:
        msg = c.get("message") or {}
        t = msg.get("content")
        if t:
            return str(t)
    candidates = resp.get("candidates") or []
    for c in candidates:
        content = c.get("content") or {}
        for p in content.get("parts") or []:
            t = p.get("text")
            if t:
                return str(t)
    return str(resp.get("text") or "")


def call_groq(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.8,
    max_tokens: int = 200,
    stream: bool = False,
) -> dict:
    if Groq is None:
        return _call_groq_http(api_key, model, system_prompt, user_prompt, temperature, max_tokens)

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=stream,
        stop=None,
    )

    if not stream:
        return completion.model_dump()

    parts = []
    for chunk in completion:
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            parts.append(text)
    return {
        "text": "".join(parts),
        "stream": True,
    }
