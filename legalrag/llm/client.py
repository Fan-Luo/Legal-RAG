from __future__ import annotations

import os
import json
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
    temperature: float = 0.86
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
        """初始化 LLM 后端（同步）"""
        if self.provider == "openai":
            api_key = os.getenv(llm_cfg.api_key_env, "")
            base_url = os.getenv(llm_cfg.base_url_env, None)

            if not api_key:
                logger.warning("[LLM] OPENAI_API_KEY 未设置，将使用降级模式。")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                logger.info(
                    "[LLM] Using OpenAI: model=%s, base_url=%s, key_env=%s",
                    self.model_name,
                    base_url or "default",
                    llm_cfg.api_key_env,
                )
        elif self.provider == "qwen-local":
            model_path = self.model_name
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

                use_cuda = torch.cuda.is_available()
                dtype = torch.float16 if use_cuda else torch.float32

                # 注意：不要让 device_map 自动把“整个模型”都 offload 到 disk
                # Kaggle/Colab 环境建议：有 GPU 则 device_map="auto"，否则显式 .to("cpu")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map="auto" if use_cuda else None,
                )

                if not use_cuda:
                    self.model.to("cpu")

                self.model.eval()
                logger.info(f"[LLM] Qwen 模型 {model_path} 加载成功")

            except Exception as e:
                logger.exception(f"[LLM] Qwen 模型加载失败，进入降级模式: {e}")
                self.tokenizer = None
                self.model = None
                self.client = None

        else:
            raise ValueError(f"Unknown provider: {self.provider}")


    # -----------------------
    # 同步接口 
    # -----------------------
    def chat(self, prompt: str) -> str:
        """
        同步生成回答。始终返回 str（适配 RagAnswer.answer: str）。
        """
        try:
            if self.provider == "openai":
                out = self._chat_openai(prompt)   # dict
            elif self.provider == "qwen-local":
                if self.model is None or self.tokenizer is None:
                    out = self._degraded_response(prompt, provider="qwen-local")
                else:
                    out = self._chat_qwen(prompt)  # dict
            else:
                out = {"text": "LLM provider not configured.", "mode": "degraded", "provider": self.provider}

        except Exception as e:
            logger.exception(f"[LLM] chat() 失败，进入降级模式: {e}")
            out = self._degraded_response(prompt, provider=self.provider)

        # 统一收敛：永远返回 str
        if isinstance(out, dict):
            txt = out.get("text")
            if isinstance(txt, str):
                return txt
            return json.dumps(out, ensure_ascii=False)

        return str(out)

    # -----------------------
    # 异步接口（可用于并发/服务端）
    # -----------------------
    async def chat_async(self, prompt: str) -> Dict[str, Any]:
        """
        兼容 async 场景：不在这里自己管理 event loop；
        直接把同步 chat() 放到线程池，避免 notebook loop 冲突。
        """
        text = await asyncio.to_thread(self.chat, prompt)
        return {"text": text, "mode": "normal", "provider": self.provider, "context_snippet": prompt[-1500:]}

    # -----------------------
    # 内部实现 
    # -----------------------
    def _chat_openai(self, prompt: str) -> Dict[str, Any]:
        if self.client is None:
            return self._degraded_response(prompt, provider="openai")

        try:
            # 简单重试
            last_err: Optional[Exception] = None
            for attempt in range(2):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "你是一名中国合同法方向的法律助手。"},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    text = resp.choices[0].message.content or ""
                    return {
                        "text": text,
                        "mode": "normal",
                        "provider": "openai",
                        "context_snippet": prompt[-1500:],
                    }
                except OpenAIError as e:
                    last_err = e
                    logger.warning(f"[LLM] OpenAI 调用失败，重试中({attempt+1}/2): {e}")
            logger.warning(f"[LLM] OpenAI 多次失败，进入降级模式: {last_err}")
            return self._degraded_response(prompt, provider="openai")
        except Exception as e:
            logger.exception(f"[LLM] OpenAI 调用异常: {e}")
            return self._degraded_response(prompt, provider="openai")

    def _chat_qwen(self, prompt: str) -> Dict[str, Any]:
        """
        本地 Qwen 同步生成。
        """
        try:
            assert self.tokenizer is not None and self.model is not None

            device = "cuda" if torch.cuda.is_available() else "cpu"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_tokens,
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 如果 decode 结果以 prompt 开头，就截掉 prompt
            text = (text[len(prompt):].strip() if text.startswith(prompt) else text.strip())

            return {
                "text": text,
                "mode": "normal",
                "provider": "qwen-local",
                "context_snippet": prompt[-1500:],
            }

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
