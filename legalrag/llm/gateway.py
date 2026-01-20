from __future__ import annotations

import time
import contextvars
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Optional
from legalrag.utils.logger import get_logger
try:
    from legalrag.llm.context import get_request_id  # type: ignore
except Exception:
    def get_request_id() -> str:
        return ""

logger = get_logger(__name__)


class LLMGateway:
    """
    Minimal LLM gateway with timeout + retry.
    Wraps an underlying client that provides chat()/chat_stream().
    """

    def __init__(
        self,
        client: Any,
        *,
        request_timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff: float = 0.6,
    ) -> None:
        self.client = client
        self.request_timeout = float(request_timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)

    def _call_with_timeout(self, fn, *args, **kwargs):
        if self.request_timeout <= 0:
            return fn(*args, **kwargs)
        with ThreadPoolExecutor(max_workers=1) as pool:
            ctx = contextvars.copy_context()
            fut = pool.submit(ctx.run, fn, *args, **kwargs)
            try:
                return fut.result(timeout=self.request_timeout)
            except TimeoutError as e:
                raise TimeoutError("llm request timeout") from e

    def chat(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, *, tag: str = ""):
        last_err = None
        for i in range(self.max_retries + 1):
            try:
                attempt_tag = f"{tag}|try={i+1}/{self.max_retries+1}" if tag else f"try={i+1}/{self.max_retries+1}"
                rid = get_request_id() or "-"
                logger.info("[LLM] request attempt %d/%d tag=%s rid=%s", i + 1, self.max_retries + 1, attempt_tag, rid)
                return self._call_with_timeout(self.client.chat, prompt=prompt, messages=messages, tag=attempt_tag)
            except Exception as e:
                last_err = e
                if i >= self.max_retries:
                    break
                time.sleep(self.retry_backoff * (2 ** i))
        raise last_err  # type: ignore[misc]

    def chat_stream(self, messages: List[Dict[str, str]]):
        return self.client.chat_stream(messages)
