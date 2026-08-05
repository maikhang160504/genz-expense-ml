from pipeline.llm_module import call_lmstudio, extract_chat_text, call_groq


def call_qwen(base_url: str, model: str, system_prompt: str, user_prompt: str, **kwargs) -> dict:
    return call_lmstudio(base_url, model, system_prompt, user_prompt, **kwargs)


__all__ = ["call_qwen", "call_lmstudio", "extract_chat_text", "call_groq"]
