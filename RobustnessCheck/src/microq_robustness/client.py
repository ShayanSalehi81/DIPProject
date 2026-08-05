"""Small, retrying client for vLLM's OpenAI-compatible multimodal endpoint."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microq_robustness.config import Settings
from microq_robustness.errors import ServerError
from microq_robustness.prompting import SYSTEM_PROMPT


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    latency_seconds: float
    attempts: int
    prompt_tokens: int | None
    completion_tokens: int | None


class VllmClient:
    """Call one-image chat completions using local base64 data URLs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._headers = {"Content-Type": "application/json"}
        if settings.api_key:
            self._headers["Authorization"] = f"Bearer {settings.api_key}"

    def assert_ready(self) -> None:
        payload = self._request_json("GET", f"{self.settings.api_base}/models", None)
        model_ids = {
            str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)
        }
        if self.settings.model_id not in model_ids:
            raise ServerError(
                f"vLLM is reachable but {self.settings.model_id!r} is not served. "
                f"Available models: {sorted(model_ids)}"
            )

    def complete(self, *, image_path: Path, user_prompt: str, seed: int) -> Completion:
        image_url = self._data_url(image_path)
        payload = {
            "model": self.settings.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "seed": seed,
        }
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = self._request_json(
                    "POST", f"{self.settings.api_base}/chat/completions", payload
                )
                content = response["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", "")) if isinstance(item, dict) else str(item)
                        for item in content
                    )
                usage = response.get("usage") or {}
                return Completion(
                    text=str(content),
                    latency_seconds=time.perf_counter() - started,
                    attempts=attempt,
                    prompt_tokens=self._optional_int(usage.get("prompt_tokens")),
                    completion_tokens=self._optional_int(usage.get("completion_tokens")),
                )
            except (ServerError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise ServerError(
            f"Completion failed after {self.settings.max_retries} attempts: {last_error}"
        )

    def _request_json(
        self, method: str, url: str, payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        encoded = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=encoded, headers=self._headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.request_timeout_seconds
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ServerError(f"vLLM returned HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ServerError(f"Could not call vLLM at {url}: {exc}") from exc
        if not isinstance(result, dict):
            raise ServerError("vLLM returned a non-object JSON response.")
        return result

    @staticmethod
    def _data_url(path: Path) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

