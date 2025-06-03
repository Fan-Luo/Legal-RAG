from __future__ import annotations

import os
from typing import Optional

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

from legalrag.config import AppConfig
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient(BaseModel):
    provider: str
    model_name: str
    max_context_tokens: int

    client: Optional[OpenAI] = None
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "LLMClient":
        llm_cfg = cfg.llm
        inst = cls(
            provider=llm_cfg.provider,
            model_name=llm_cfg.model,
            max_context_tokens=llm_cfg.max_context_tokens,
        )
        inst._init_backend(llm_cfg)
        return inst

    def _init_backend(self, llm_cfg):
        if self.provider == "openai":
            api_key = os.getenv(llm_cfg.api_key_env, "")
            base_url = os.getenv(llm_cfg.base_url_env, None)
            if not api_key:
                logger.warning("[LLM] OPENAI_API_KEY 未设置，将使用降级模式。")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                logger.info("[LLM] OpenAI client 初始化完成")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def chat(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._chat_openai(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _chat_openai(self, prompt: str) -> str:
        if self.client is None:
            return (
                "【注意】OpenAI API 未配置或不可用，当前只能展示检索到的条文。"
            )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一名中国合同法方向的法律助手。"},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.exception(f"[LLM] OpenAI 调用失败: {e}")
            return (
                "【注意】OpenAI 调用失败，当前只能展示检索到的条文。"
            )