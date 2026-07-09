from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv

from agent.models import ModelResponse, Usage


class QwenClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 120,
        enable_thinking: bool | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = (base_url or os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.model = model or os.getenv("QWEN_MODEL") or "qwen3.6-plus"
        self.timeout_seconds = timeout_seconds
        if enable_thinking is None:
            enable_thinking = _env_flag("QWEN_ENABLE_THINKING", default=False)
        self.enable_thinking = enable_thinking
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for Qwen calls")
        if not self.model.startswith("qwen"):
            raise RuntimeError("QWEN_MODEL must be a Qwen model name")

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "enable_thinking": self.enable_thinking,
        }
        raw = self._post_json(f"{self.base_url}/chat/completions", payload)
        content = _extract_content(raw)
        parsed = _parse_json_content(content)
        return ModelResponse(
            parsed=parsed,
            content=content,
            usage=Usage.from_dict(raw.get("usage")),
            raw=raw,
        )

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Qwen HTTP error {exc.code}: {detail}") from exc
        return json.loads(data)


def _extract_content(raw: dict[str, Any]) -> str:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Qwen response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Qwen response content is empty")
    return content.strip()


def _parse_json_content(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qwen response is not valid JSON: {content[:200]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Qwen JSON response must be an object")
    return parsed


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
