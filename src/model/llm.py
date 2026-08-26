"""Hugging Face causal LLM wrapper for baseline / fine-tuned generation."""

from __future__ import annotations

import logging
from typing import Any

from src.utils.config import load_yaml
from src.utils.paths import resolve_path

logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    """Thin generation wrapper around an instruction-tuned HF model.

    Optional LoRA adapter path enables System C (fine-tuned + RAG).
    """

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        adapter_path: str | None = None,
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    @classmethod
    def from_configs(
        cls,
        model_cfg: dict[str, Any] | None = None,
        *,
        adapter_path: str | None = None,
    ) -> "HuggingFaceLLM":
        model_cfg = model_cfg or load_yaml("./config/model.yaml")
        selected = model_cfg.get("selected", {})
        quant = model_cfg.get("quantization", {})
        inference = model_cfg.get("inference", {})
        return cls(
            model_id=str(selected.get("model_id")),
            revision=selected.get("revision"),
            trust_remote_code=bool(selected.get("trust_remote_code", False)),
            load_in_4bit=bool(quant.get("load_in_4bit", True)),
            max_new_tokens=int(inference.get("max_new_tokens", 512)),
            temperature=float(inference.get("temperature", 0.2)),
            top_p=float(inference.get("top_p", 0.9)),
            do_sample=bool(inference.get("do_sample", True)),
            repetition_penalty=float(inference.get("repetition_penalty", 1.05)),
            adapter_path=adapter_path,
        )

    def _cuda_available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _bitsandbytes_importable() -> bool:
        try:
            import bitsandbytes  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def _load_unquantized(self, model_kwargs: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        model_kwargs.pop("quantization_config", None)
        if self._cuda_available():
            model_kwargs["device_map"] = self.device_map
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, dtype=torch.float16, **model_kwargs
                )
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, torch_dtype=torch.float16, **model_kwargs
                )
            return
        model_kwargs["torch_dtype"] = torch.float32
        model_kwargs["device_map"] = None
        logger.warning(
            "4-bit CUDA load unavailable; loading %s on CPU (slower).",
            self.model_id,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **model_kwargs
        )
        self.model.to("cpu")

    def load(self) -> None:
        """Load tokenizer + model (lazy; call once)."""
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM %s", self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_4bit = (
            self.load_in_4bit
            and self._cuda_available()
            and self._bitsandbytes_importable()
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "revision": self.revision,
        }

        if use_4bit:
            try:
                from src.finetuning.train import ensure_triton_ops_stub

                ensure_triton_ops_stub()
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["device_map"] = self.device_map
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, **model_kwargs
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "4-bit load failed (%s); using fp16/CPU fallback.",
                    exc,
                )
                use_4bit = False
                model_kwargs.pop("quantization_config", None)

        if not use_4bit:
            self._load_unquantized(model_kwargs)

        if self.adapter_path:
            from src.finetuning.train import (
                ensure_triton_ops_stub,
                patch_incompatible_torchao,
            )
            from peft import PeftModel

            ensure_triton_ops_stub()
            patch_incompatible_torchao()
            adapter = resolve_path(self.adapter_path)
            logger.info("Loading LoRA adapter from %s", adapter)
            self.model = PeftModel.from_pretrained(self.model, str(adapter))

        self.model.eval()

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate an assistant reply from chat messages."""
        import torch

        self.load()
        assert self.tokenizer is not None and self.model is not None

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": max(self.temperature, 1e-5),
                    "top_p": self.top_p,
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()
