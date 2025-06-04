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
        elif self.provider == "qwen-local":
            model_path = self.model_name
            logger.info(f"[LLM] 加载本地 Qwen 模型：{model_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                if torch.cuda.is_available():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                    ).to("cpu")
                self.model.eval()
                logger.info("[LLM] Qwen 模型加载成功")
            except Exception as e:
                logger.exception(f"[LLM] Qwen 加载失败，进入降级模式: {e}")
                self.tokenizer = None
                self.model = None
                self.client = None
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def chat(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._chat_openai(prompt)
        elif self.provider == "qwen-local":
            if self.model is None or self.tokenizer is None:
                return (
                    "【注意】本地 Qwen 模型未成功加载，当前为降级演示模式。\n\n"
                    "以下为系统整理的参考语境，请结合实际法律条文自行判断：\n\n"
                    + prompt[-1500:]
                )
            return self._chat_qwen(prompt)
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

    def _chat_qwen(self, prompt: str) -> str:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_tokens,
        ).to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt) :].strip() or text.strip()
