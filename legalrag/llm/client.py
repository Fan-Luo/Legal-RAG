from __future__ import annotations
import os
import asyncio
from typing import Optional, Dict, Any

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI, OpenAIError

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

    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "LLMClient":
        llm_cfg = cfg.llm
        inst = cls(
            provider=llm_cfg.provider,
            model_name=llm_cfg.model,
            max_context_tokens=llm_cfg.max_context_tokens,
            max_new_tokens=getattr(llm_cfg, "max_new_tokens", 512),
            temperature=getattr(llm_cfg, "temperature", 0.3),
            top_p=getattr(llm_cfg, "top_p", 0.9),
        )
        inst._init_backend(llm_cfg)
        return inst

    def _init_backend(self, llm_cfg):
        """初始化 LLM 后端"""
        if self.provider == "openai":
            api_key = os.getenv(llm_cfg.api_key_env, "")
            base_url = os.getenv(llm_cfg.base_url_env, None)
            if not api_key:
                logger.warning("[LLM] OPENAI_API_KEY 未设置，将使用降级模式。")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                logger.info("[LLM] OpenAI client 初始化完成")
        elif self.provider == "qwen-local":
            model_path = self.model_name
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, trust_remote_code=True, torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None
                )
                self.model.eval()
                logger.info("[LLM] Qwen 模型加载成功")
            except Exception as e:
                logger.exception(f"[LLM] Qwen 模型加载失败，进入降级模式: {e}")
                self.tokenizer = None
                self.model = None
                self.client = None
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # -----------------------
    # 异步接口
    # -----------------------
    async def chat_async(self, prompt: str) -> Dict[str, Any]:
        """异步生成回答"""
        if self.provider == "openai":
            return await self._chat_openai(prompt)
        elif self.provider == "qwen-local":
            if self.model is None or self.tokenizer is None:
                return self._degraded_response(prompt)
            return await self._chat_qwen(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # -----------------------
    # 同步兼容接口
    # -----------------------
    def chat(self, prompt: str) -> Dict[str, Any]:
        return asyncio.run(self.chat_async(prompt))

    # -----------------------
    # 内部方法
    # -----------------------
    async def _chat_openai(self, prompt: str) -> Dict[str, Any]:
        if self.client is None:
            return self._degraded_response(prompt, provider="openai")
        try:
            for attempt in range(2):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "你是一名中国合同法方向的法律助手。"},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    text = resp.choices[0].message.content
                    return {"text": text, "mode": "normal", "provider": self.provider, "context_snippet": prompt[-1500:]}
                except OpenAIError as e:
                    logger.warning(f"[LLM] OpenAI 调用失败，重试中: {e}")
                    await asyncio.sleep(1)
            return self._degraded_response(prompt, provider="openai")
        except Exception as e:
            logger.exception(f"[LLM] OpenAI 调用异常: {e}")
            return self._degraded_response(prompt, provider="openai")

    async def _chat_qwen(self, prompt: str) -> Dict[str, Any]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=self.max_context_tokens
            ).to(device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            text = text[len(prompt):].strip() or text.strip()
            return {"text": text, "mode": "normal", "provider": self.provider, "context_snippet": prompt[-1500:]}
        except Exception as e:
            logger.exception(f"[LLM] 本地 Qwen 生成失败，进入降级模式: {e}")
            return self._degraded_response(prompt, provider="qwen-local")

    def _degraded_response(self, prompt: str, provider: Optional[str] = None) -> Dict[str, Any]:
        provider = provider or self.provider
        return {
            "text": (
                f"【降级模式】{provider} 模型不可用或生成失败。\n\n"
                "当前系统仅展示检索到的条文，LLM 问答功能暂不可用。\n\n"
                "提示：最近文本上下文（长度限制 1500 字符）：\n\n" + prompt[-1500:]
            ),
            "mode": "degraded",
            "provider": provider,
            "context_snippet": prompt[-1500:],
        }
