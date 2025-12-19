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
    temperature: float = 0.5
    top_p: float = 0.9
    repetition_penalty: float = 1.12
    no_repeat_ngram_size: int = 0

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


    @classmethod
    def from_config_with_key(cls, cfg: AppConfig, openai_key: str | None):
        llm_cfg = cfg.llm

        # If user provided a key, force OpenAI for this instance only.
        provider = "openai" if (openai_key and openai_key.strip()) else llm_cfg.provider

        if  provider == "openai":
            # Pick model name: prefer OPENAI_MODEL env if present, else cfg.llm.model
            openai_model = (os.getenv("OPENAI_MODEL", "") or "").strip()
            cfg_model = (getattr(llm_cfg, "openai_model", "") or "").strip()
            model_name = openai_model or cfg_model or "gpt-4o-mini"
        elif rovider == "qwen-local":
            cfg_model = (getattr(llm_cfg, "qwen_model", "") or "").strip()
            model_name = cfg_model or "Qwen/Qwen2.5-3B-Instruct"
        else:
            model_name = (getattr(llm_cfg, "model", "") or "").strip()

        inst = cls(
            provider=provider,
            model_name=model_name,
            max_context_tokens=llm_cfg.max_context_tokens,
            max_new_tokens=getattr(llm_cfg, "max_new_tokens", 512),
            temperature=getattr(llm_cfg, "temperature", 0.3),
            top_p=getattr(llm_cfg, "top_p", 0.9),
        )
        logger.info(f"[LLM] override init provider={inst.provider} model_name={inst.model_name!r} env_OPENAI_MODEL={os.getenv('OPENAI_MODEL','')!r}")
        inst._init_backend(llm_cfg, override_openai_key=openai_key)
        return inst

    def _init_backend(self, llm_cfg, override_openai_key: str | None = None):
        """初始化 LLM 后端（同步）"""
        if self.provider == "openai":
            api_key = override_openai_key or os.getenv(llm_cfg.api_key_env, "")
            base_url = os.getenv(llm_cfg.base_url_env, None)

            if not api_key:
                logger.warning("[LLM] No OpenAI API key available.")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                logger.info(
                    "[LLM] Using OpenAI: model=%s, base_url=%s, key_source=%s",
                    self.model_name,
                    base_url or "default",
                    "user" if override_openai_key else "env",
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

        elif self.provider == "disabled":
            # No backend. Server will require user-provided OpenAI key per request.
            self.client = None
            self.model = None
            self.tokenizer = None
            logger.info("[LLM] provider=disabled: backend not initialized")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


    # -----------------------
    # 同步接口（支持 prompt 或 messages）
    # -----------------------
    def chat(self, prompt: str | None = None, messages: List[Dict[str, str]] | None = None) -> str:
        """
        同步生成回答。
        - 兼容旧用法：chat(prompt="...")
        - 新用法：chat(messages=[{"role":"system","content":"..."}, {"role":"user","content":"..."}])
        """
        try:
            if messages is None:
                if prompt is None:
                    raise ValueError("Either prompt or messages must be provided.")
                messages = [{"role": "user", "content": prompt}]
            else:
                prompt = prompt or ""

            if self.provider == "openai":
                logger.info(f"[LLM] OpenAI request model={self.model_name!r}")
                if not self.model_name:
                    raise ValueError("OpenAI model is empty. Set OPENAI_MODEL or configure cfg.llm.model.")
                out = self._chat_openai_messages(messages)
            elif self.provider == "qwen-local":
                if self.model is None or self.tokenizer is None:
                    out = self._degraded_response(prompt or "", provider="qwen-local")
                else:
                    out = self._chat_qwen_messages(messages)
            elif self.provider == "disabled":
                return self._degraded_response(prompt or "", provider="disabled")["text"]
            else:
                out = {"text": "LLM provider not configured.", "mode": "degraded", "provider": self.provider}

        except Exception as e:
            logger.exception(f"[LLM] chat() 失败: {e}")

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
        text = await asyncio.to_thread(self.chat, prompt)
        return {"text": text, "mode": "normal", "provider": self.provider, "context_snippet": prompt[-1500:]}

    # -----------------------
    # OpenAI（messages 原生）
    # -----------------------
    def _chat_openai_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        if self.client is None:
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break
            return self._degraded_response(last_user, provider="openai")

        def _last_user_text() -> str:
            for m in reversed(messages):
                if m.get("role") == "user":
                    return m.get("content", "") or ""
            return ""

        def _is_restricted_sampling_model(model: str) -> bool:
            """
            GPT-5 / reasoning-like models often reject non-default sampling params
            (e.g., temperature != 1). To be safe, omit sampling params entirely.
            """
            m = (model or "").lower().strip()
            # Cover common GPT-5 names and likely reasoning families.
            return (
                m.startswith("gpt-5")
                or m.startswith("o1")
                or m.startswith("o3")
                or "thinking" in m
            )

        try:
            last_err: Optional[Exception] = None

            for attempt in range(2):
                try:
                    # Base required args
                    kwargs: Dict[str, Any] = {
                        "model": self.model_name,
                        "messages": messages,
                    }

                    # Only pass sampling params if model family supports it.
                    if not _is_restricted_sampling_model(self.model_name):
                        if self.temperature is not None:
                            kwargs["temperature"] = self.temperature
                        if self.top_p is not None:
                            kwargs["top_p"] = self.top_p

                    logger.info(
                        "[LLM] OpenAI request model=%r kwargs_keys=%s",
                        self.model_name,
                        sorted(kwargs.keys()),
                    )

                    resp = self.client.chat.completions.create(**kwargs)
                    text = resp.choices[0].message.content or ""

                    last_user = _last_user_text()
                    return {
                        "text": text,
                        "mode": "normal",
                        "provider": "openai",
                        "context_snippet": last_user[-1500:],
                    }

                except OpenAIError as e:
                    last_err = e
                    logger.warning(f"[LLM] OpenAI 调用失败，重试中({attempt+1}/2): {e}")

            logger.warning(f"[LLM] OpenAI 多次失败，进入降级模式: {last_err}")
            return self._degraded_response(_last_user_text(), provider="openai")

        except Exception as e:
            logger.exception(f"[LLM] OpenAI 调用异常: {e}")
            return self._degraded_response(_last_user_text(), provider="openai")

    # -----------------------
    # Qwen-local（messages -> chat_template -> generate）
    # -----------------------
    def _chat_qwen_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            assert self.tokenizer is not None and self.model is not None

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 关键：用 chat_template 把 messages 渲染成模型可用的 prompt
            if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # fallback：拼接（效果弱于 chat_template，但至少可运行）
                rendered = "\n\n".join([f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages]) + "\n\nASSISTANT:"

            inputs = self.tokenizer(
                rendered,
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
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                )

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 常见情况：decode 结果含 prompt，截断掉 prompt 前缀
            text = (text[len(rendered):].strip() if text.startswith(rendered) else text.strip())

            # snippet：最后一个 user
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break

            return {
                "text": text,
                "mode": "normal",
                "provider": "qwen-local",
                "context_snippet": last_user[-1500:],
            }

        except Exception as e:
            logger.exception(f"[LLM] 本地 Qwen 生成失败，进入降级模式: {e}")
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break
            return self._degraded_response(last_user, provider="qwen-local")

    def _degraded_response(self, prompt: str, provider: Optional[str] = None) -> Dict[str, Any]:
        provider = provider or self.provider
        return {
            "text": (
                f"【降级模式】模型不可用, LLM 问答功能暂不可用。\n\n"
                "当前系统仅展示检索到的条文。\n\n"
            ),
            "mode": "degraded",
            "provider": provider,
            "context_snippet": prompt[-1500:],
        }
