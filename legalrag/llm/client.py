from __future__ import annotations

import os
import json
import asyncio
from typing import Optional, Dict, Any, List

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI, OpenAIError

from legalrag.config import AppConfig
from legalrag.utils.logger import get_logger
from typing import AsyncIterator
from transformers import TextIteratorStreamer
import threading
import queue 


logger = get_logger(__name__)

def _is_restricted_sampling_model(model: str) -> bool:
    """Return True if the model is known to reject non-default sampling params.

    Some OpenAI model families (e.g., GPT-5 / reasoning-style) may reject
    temperature/top_p values other than defaults. In those cases we omit
    sampling parameters entirely to avoid 400 errors.
    """
    m = (model or "").lower().strip()
    return (
        m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or "thinking" in m
    )

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
    _default_instance: ClassVar[Optional["LLMClient"]] = None
    _instances_by_key: ClassVar[Dict[str, "LLMClient"]] = {}

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "LLMClient":

        # Singleton: return existing instance if already created 
        if cls._default_instance is not None: 
            return cls._default_instance

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
        cls._default_instance = inst
        return inst


    @classmethod
    def from_config_with_key(cls, cfg: AppConfig, openai_key: str | None):

        key = openai_key or "__no_key__" 
        if key in cls._instances_by_key: 
            return cls._instances_by_key[key]

        llm_cfg = cfg.llm

        # If user provided a key, force OpenAI for this instance only.
        provider = "openai" if (openai_key and openai_key.strip()) else llm_cfg.provider

        if  provider == "openai":
            # Pick model name: prefer OPENAI_MODEL env if present, else cfg.llm.model
            openai_model = (os.getenv("OPENAI_MODEL", "") or "").strip()
            cfg_model = (getattr(llm_cfg, "openai_model", "") or "").strip()
            model_name = openai_model or cfg_model or "gpt-4o-mini"
        elif provider == "qwen-local":
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
        cls._instances_by_key[key] = inst
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

                # 不让 device_map 自动把“整个模型”都 offload 到 disk
                # 有 GPU 则 device_map="auto"，否则显式 .to("cpu")
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
                # logger.info(f"[chat]: messages" )
            else:
                prompt = prompt or ""
                # logger.info(f"[chat]: {prompt}" )
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
        logger.info(f"[_chat_openai_messages]")
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

            # 用 chat_template 把 messages 渲染成模型可用的 prompt
            if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                # logger.info(f"[_chat_qwen_messages] apply_chat_template")
            else:
                # fallback：拼接 
                rendered = "\n\n".join([f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages]) + "\n\nASSISTANT:"
                logger.info(f"[_chat_qwen_messages] fallback")
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

            # decode 结果含 prompt，截断掉 prompt 前缀
            input_len = inputs["input_ids"].shape[-1]
            new_tokens = outputs[0][input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True) 

            # 最后一个 user
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

    async def chat_stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """
        异步流式生成接口
        - OpenAI: 使用官方 stream=True
        - Qwen-local: 使用 TextIteratorStreamer + 后台线程
        """
        if self.provider == "openai":
            if self.client is None:
                yield "【错误】OpenAI API Key 未配置"
                return
            logger.info(f"[chat_stream] openai 模型")
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                }
                if not _is_restricted_sampling_model(self.model_name):
                    if self.temperature is not None:
                        kwargs["temperature"] = self.temperature
                    if self.top_p is not None:
                        kwargs["top_p"] = self.top_p

                loop = asyncio.get_running_loop()
                q: asyncio.Queue[str | None] = asyncio.Queue()
                err: dict = {"e": None}

                def _worker():
                    try:
                        stream = self.client.chat.completions.create(**kwargs)  # 同步迭代器
                        for chunk in stream:
                            content = chunk.choices[0].delta.content
                            if content:
                                asyncio.run_coroutine_threadsafe(q.put(content), loop)
                    except Exception as e:
                        err["e"] = e
                    finally:
                        asyncio.run_coroutine_threadsafe(q.put(None), loop)

                threading.Thread(target=_worker, daemon=True).start()

                while True:
                    item = await q.get()
                    if item is None:
                        break
                    yield item

                if err["e"] is not None:
                    yield f"\n【流式错误】OpenAI: {type(err['e']).__name__}: {err['e']}"
                return

            except Exception as e:
                logger.exception("[LLM] OpenAI streaming failed")
                yield f"\n【流式错误】OpenAI: {str(e)}"

        elif self.provider == "qwen-local":
            if self.model is None or self.tokenizer is None:
                yield "【错误】本地 Qwen 模型未加载"
                return
            logger.info(f"[chat_stream] qwen 模型")
            try:
                assert self.tokenizer is not None and self.model is not None

                device = "cuda" if torch.cuda.is_available() else "cpu"

                # 用 chat_template 把 messages 渲染成模型可用的 prompt
                if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
                    rendered = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    logger.info(f"[chat_stream] apply_chat_template")
                else:
                    # fallback：拼接 
                    rendered = "\n\n".join([f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages]) + "\n\nASSISTANT:"
                    logger.info(f"[chat_stream] fallback")
                inputs = self.tokenizer(
                    rendered,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_context_tokens,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logger.info(f"[chat_stream] inputs.keys() = %s", inputs.keys())
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,           # 不输出原始 prompt
                    skip_special_tokens=True,
                    timeout=30.0,
                )

                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "repetition_penalty": self.repetition_penalty,
                    "no_repeat_ngram_size": self.no_repeat_ngram_size,
                    "streamer": streamer,
                }
                stop_event = threading.Event()

                gen_exc = {"err": None}

                def _run_generate():
                    try:
                        self.model.generate(**generation_kwargs)
                    except Exception as e:
                        gen_exc["err"] = e
                        logger.exception("[qwen-local] model.generate failed in worker thread")
                    finally:
                        logger.info("[chat_stream] finally")
                        # 如果 generate 异常退出，end() 不会被框架调用，streamer 会一直等
                        try:
                            if hasattr(streamer, "end"):
                                logger.info("[chat_stream] streamer.end()")
                                streamer.end()
                        except Exception:
                            pass
                        stop_event.set()
                        logger.info("[chat_stream] stop_event.set()")

                thread = threading.Thread(target=_run_generate, daemon=True)
                thread.start()

                async for new_text in self._async_streamer_iterator(streamer, stop_event):
                    # logger.info("[chat_stream] =%s", new_text)
                    if new_text:
                        yield new_text

                thread.join(timeout=1.0)
                if gen_exc["err"] is not None:
                    yield f"\n【流式错误】Qwen(generate): {type(gen_exc['err']).__name__}: {gen_exc['err']}"
            except Exception as e:
                logger.exception("[LLM] Qwen local streaming failed")
                yield f"\n【流式错误】Qwen: {str(e)}"

        else:
            yield "【错误】当前 provider 不支持 streaming"
            return

    # 把 TextIteratorStreamer 转成 async iterator
    async def _async_streamer_iterator(self, streamer, stop_event: threading.Event):
        logger.info("[_async_streamer_iterator]")

        stop_sig = getattr(streamer, "stop_signal", None) 

        logger.info("streamer stop signal: %r", stop_sig)

        while True:
            try:
                text = streamer.text_queue.get(timeout=0.1)

                if stop_sig is not None and text is stop_sig:
                    break

                if text:
                    yield text

            except queue.Empty:
                if stop_event.is_set():
                    break
                await asyncio.sleep(0.01)
            except Exception:
                logger.exception("[_async_streamer_iterator] failed")
                break

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