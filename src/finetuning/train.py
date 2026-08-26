"""QLoRA fine-tuning for DataPilot AI (Dataset B → LoRA adapter).

Designed for Google Colab T4. Does not fabricate metrics — summary JSON is written
only from actual Trainer log history after a real run.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.finetuning.dataset import load_jsonl
from src.finetuning.formatting import row_to_text
from src.utils.config import load_yaml
from src.utils.paths import project_root, resolve_path

logger = logging.getLogger(__name__)


def ensure_triton_ops_stub() -> None:
    """Colab Triton removed ``triton.ops``; leftover bitsandbytes still imports it.

    Call this *before* importing transformers/peft/trl.
    """
    import sys
    import types

    try:
        from triton.ops.matmul_perf_model import early_config_prune  # noqa: F401
        return
    except Exception:
        pass

    if "triton" not in sys.modules:
        sys.modules["triton"] = types.ModuleType("triton")
    ops = types.ModuleType("triton.ops")
    matmul = types.ModuleType("triton.ops.matmul_perf_model")
    matmul.early_config_prune = lambda *_a, **_k: []
    matmul.estimate_matmul_time = lambda *_a, **_k: 0
    ops.matmul_perf_model = matmul
    sys.modules["triton.ops"] = ops
    sys.modules["triton.ops.matmul_perf_model"] = matmul
    try:
        import triton

        triton.ops = ops
    except Exception:  # noqa: BLE001
        pass
    logger.info("Installed triton.ops compatibility stub (Colab / bitsandbytes).")


def patch_incompatible_torchao() -> None:
    """PEFT raises if torchao is installed but older than 0.16 (common on Colab).

    Standard LoRA on ``nn.Linear`` does not need torchao, so treat old versions as absent.
    """
    import sys

    def _is_torchao_ok() -> bool:
        try:
            import importlib.metadata as md
            from packaging.version import Version

            return Version(md.version("torchao")) > Version("0.16.0")
        except Exception:
            return False

    try:
        import peft.import_utils as iu
    except Exception:  # noqa: BLE001
        return
    iu.is_torchao_available = _is_torchao_ok
    for name, mod in list(sys.modules.items()):
        if name.startswith("peft") and hasattr(mod, "is_torchao_available"):
            setattr(mod, "is_torchao_available", _is_torchao_ok)
    logger.info("Patched PEFT torchao availability check for this runtime.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_model_config(path: str | Path = "./config/model.yaml") -> dict[str, Any]:
    return load_yaml(path)


def resolve_adapter_run_dir(base_dir: str | Path, run_name: str | None = None) -> Path:
    base = resolve_path(base_dir)
    name = run_name or f"qwen25_1p5b_qlora_{_utc_now()}"
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_hf_dataset(rows: list[dict[str, Any]], tokenizer: Any | None = None):
    """Build a Hugging Face Dataset with a ``text`` column for SFTTrainer."""
    from datasets import Dataset

    texts = [row_to_text(row, tokenizer=tokenizer) for row in rows]
    return Dataset.from_dict({"text": texts})


def dry_run_report(
    *,
    model_cfg: dict[str, Any],
    train_path: Path,
    val_path: Path,
    sample_n: int = 2,
) -> dict[str, Any]:
    """Validate data + formatting without loading the full LLM weights."""
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    samples = [row_to_text(r, tokenizer=None) for r in train_rows[:sample_n]]

    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda = False

    selected = model_cfg.get("selected", {})
    training = model_cfg.get("training", {})
    return {
        "ok": True,
        "mode": "dry_run",
        "model_id": selected.get("model_id"),
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "cuda_available": cuda,
        "planned_epochs": training.get("num_epochs"),
        "max_epochs_cap": training.get("max_epochs_cap"),
        "early_stopping_patience": training.get("early_stopping_patience"),
        "per_device_train_batch_size": training.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "max_seq_length": training.get("max_seq_length"),
        "sample_formatted_preview": samples,
        "notes": [
            "Dry-run does not train or download model weights.",
            "Run on Colab T4 (or local CUDA) without --dry-run to produce an adapter.",
            "Do not invent training metrics; they appear only after a real run.",
        ],
    }


def resolve_use_4bit(*, no_4bit: bool, model_cfg: dict[str, Any]) -> bool:
    """Return True only when 4-bit QLoRA can actually load on this runtime.

    Current Colab (PyTorch cu128 + new Triton) often has no GPU bitsandbytes
    wheel and crashes on ``triton.ops``. Qwen2.5-1.5B fits T4 in fp16 LoRA.
    """
    if no_4bit:
        logger.info("4-bit disabled by flag; using fp16 LoRA.")
        return False
    if not bool(model_cfg.get("quantization", {}).get("load_in_4bit", True)):
        return False
    try:
        _bits_and_bytes_config(model_cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "4-bit QLoRA unavailable (%s). Falling back to fp16 LoRA. "
            "This is expected on Colab CUDA 12.8 / current Triton.",
            exc,
        )
        return False
    return True


def _bits_and_bytes_config(model_cfg: dict[str, Any]):
    import torch
    from transformers import BitsAndBytesConfig

    quant = model_cfg.get("quantization", {})
    compute = str(quant.get("bnb_4bit_compute_dtype", "float16")).lower()
    dtype = torch.float16 if compute in {"float16", "fp16"} else torch.bfloat16
    return BitsAndBytesConfig(
        load_in_4bit=bool(quant.get("load_in_4bit", True)),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type=str(quant.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
    )


def _lora_config(model_cfg: dict[str, Any]):
    from peft import LoraConfig, TaskType

    lora = model_cfg.get("lora", {})
    task = str(lora.get("task_type", "CAUSAL_LM"))
    return LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("lora_alpha", 32)),
        lora_dropout=float(lora.get("lora_dropout", 0.05)),
        bias=str(lora.get("bias", "none")),
        task_type=TaskType[task] if task in TaskType.__members__ else TaskType.CAUSAL_LM,
        target_modules=list(
            lora.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )


def _supported_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only constructor args accepted by this transformers version."""
    import inspect

    params = set(inspect.signature(cls.__init__).parameters)
    params.discard("self")
    kept = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        logger.info("Ignoring unsupported %s fields: %s", getattr(cls, "__name__", cls), dropped)
    return kept


def _training_arguments(
    model_cfg: dict[str, Any],
    output_dir: Path,
    *,
    use_4bit: bool,
    max_seq_length: int = 1024,
):
    """Build SFTConfig (current TRL) or TrainingArguments (older stacks)."""
    from transformers import TrainingArguments

    t = model_cfg.get("training", {})
    epochs = float(t.get("num_epochs", 3))
    cap = float(t.get("max_epochs_cap", 25))
    if epochs > cap:
        logger.warning("num_epochs %.1f exceeds cap %.1f; clamping.", epochs, cap)
        epochs = cap

    fp16 = bool(t.get("fp16", True))
    bf16 = bool(t.get("bf16", False))
    # Prefer bf16 on Ampere+ if requested; T4 is Turing → fp16.
    try:
        import torch

        if bf16 and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                logger.info("GPU compute capability < 8; using fp16 instead of bf16.")
                bf16 = False
                fp16 = True
    except Exception:  # noqa: BLE001
        pass

    warmup = float(t.get("warmup_ratio", 0.03))
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": int(t.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(t.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(t.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(t.get("learning_rate", 2e-4)),
        "lr_scheduler_type": str(t.get("lr_scheduler_type", "cosine")),
        # v4: warmup_ratio. v5: warmup_steps accepts a float ratio.
        "warmup_ratio": warmup,
        "warmup_steps": warmup,
        "weight_decay": float(t.get("weight_decay", 0.01)),
        "logging_steps": int(t.get("logging_steps", 10)),
        "eval_strategy": str(t.get("eval_strategy", "epoch")),
        "evaluation_strategy": str(t.get("eval_strategy", "epoch")),
        "save_strategy": str(t.get("save_strategy", "epoch")),
        "save_total_limit": int(t.get("save_total_limit", 2)),
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": bool(t.get("gradient_checkpointing", True)),
        # paged_adamw_8bit requires a GPU bitsandbytes build.
        "optim": (
            str(t.get("optim", "paged_adamw_8bit"))
            if use_4bit
            else "adamw_torch"
        ),
        "report_to": t.get("report_to", "none"),
        "logging_dir": str(resolve_path(t.get("logging_dir", "./experiments/logs/training"))),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": int(model_cfg.get("reproducibility", {}).get("seed", 42)),
        "remove_unused_columns": False,
        # TRL SFTConfig fields (dropped automatically on plain TrainingArguments).
        "dataset_text_field": "text",
        "packing": False,
        "max_seq_length": max_seq_length,
        "max_length": max_seq_length,
    }
    import inspect

    try:
        from trl import SFTConfig

        params = inspect.signature(SFTConfig.__init__).parameters
        if "warmup_ratio" in params:
            kwargs.pop("warmup_steps", None)
        elif "warmup_steps" in params:
            kwargs.pop("warmup_ratio", None)
        logger.info("Using TRL SFTConfig")
        return SFTConfig(**_supported_kwargs(SFTConfig, kwargs))
    except Exception as exc:  # noqa: BLE001
        logger.info("SFTConfig unavailable (%s); using TrainingArguments", exc)
        params = inspect.signature(TrainingArguments.__init__).parameters
        if "warmup_ratio" in params:
            kwargs.pop("warmup_steps", None)
        else:
            kwargs.pop("warmup_ratio", None)
        return TrainingArguments(**_supported_kwargs(TrainingArguments, kwargs))


def _make_sft_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    val_ds: Any,
    peft_config: Any,
    args: Any,
    callbacks: list[Any],
    max_seq: int,
) -> Any:
    """Construct SFTTrainer across TRL 0.9–current APIs."""
    from trl import SFTTrainer

    patch_incompatible_torchao()

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "callbacks": callbacks,
        "dataset_text_field": "text",
        "max_seq_length": max_seq,
        "max_length": max_seq,
        "packing": False,
    }
    return SFTTrainer(**_supported_kwargs(SFTTrainer, trainer_kwargs))


def _extract_metrics(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull real train/eval losses from Trainer logs (no invention)."""
    train_losses = [x["loss"] for x in log_history if "loss" in x and "eval_loss" not in x]
    eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]
    last_train = next((x for x in reversed(log_history) if "train_runtime" in x), None)
    return {
        "train_loss_steps": train_losses,
        "eval_loss_by_epoch": eval_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "best_eval_loss": min(eval_losses) if eval_losses else None,
        "last_eval_loss": eval_losses[-1] if eval_losses else None,
        "train_runtime_seconds": None if last_train is None else last_train.get("train_runtime"),
        "train_samples_per_second": None
        if last_train is None
        else last_train.get("train_samples_per_second"),
        "log_history": log_history,
    }


def run_qlora_training(
    *,
    model_cfg: dict[str, Any] | None = None,
    train_path: str | Path = "./data/finetuning/train.jsonl",
    val_path: str | Path = "./data/finetuning/validation.jsonl",
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    no_4bit: bool = False,
) -> dict[str, Any]:
    """Execute PEFT SFT (QLoRA 4-bit when available, else fp16 LoRA)."""
    ensure_triton_ops_stub()
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
    )

    model_cfg = model_cfg or load_model_config()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for QLoRA training. Use Google Colab T4 "
            "(notebooks/04_qlora_finetune.ipynb) or a local NVIDIA GPU."
        )

    selected = model_cfg.get("selected", {})
    model_id = str(selected.get("model_id"))
    training = model_cfg.get("training", {})
    paths = model_cfg.get("paths", {})

    train_file = resolve_path(train_path)
    val_file = resolve_path(val_path)
    train_rows = load_jsonl(train_file)
    val_rows = load_jsonl(val_file)
    if max_train_samples is not None:
        train_rows = train_rows[: max(1, max_train_samples)]
    if max_val_samples is not None:
        val_rows = val_rows[: max(1, max_val_samples)]

    base_out = output_dir or paths.get("adapter_dir") or training.get("output_dir")
    run_dir = resolve_adapter_run_dir(base_out, run_name=run_name)
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    use_4bit = resolve_use_4bit(no_4bit=no_4bit, model_cfg=model_cfg)
    method = "QLoRA-4bit" if use_4bit else "LoRA-fp16"
    logger.info("Loading tokenizer/model %s (%s)", model_id, method)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=selected.get("revision"),
        trust_remote_code=bool(selected.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs: dict[str, Any] = {
        "revision": selected.get("revision"),
        "trust_remote_code": bool(selected.get("trust_remote_code", False)),
        "device_map": "auto",
    }
    if use_4bit:
        from peft import prepare_model_for_kbit_training

        load_kwargs["quantization_config"] = _bits_and_bytes_config(model_cfg)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, **load_kwargs
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, **load_kwargs
            )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    elif training.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if training.get("gradient_checkpointing", True):
        model.config.use_cache = False

    train_ds = build_hf_dataset(train_rows, tokenizer=tokenizer)
    val_ds = build_hf_dataset(val_rows, tokenizer=tokenizer)
    patience = int(training.get("early_stopping_patience", 2))
    max_seq = int(training.get("max_seq_length", 1024))
    args = _training_arguments(
        model_cfg, run_dir, use_4bit=use_4bit, max_seq_length=max_seq
    )
    trainer = _make_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        peft_config=_lora_config(model_cfg),
        args=args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        max_seq=max_seq,
    )

    t0 = time.perf_counter()
    logger.info("Starting %s training → %s", method, run_dir)
    train_result = trainer.train()
    elapsed = time.perf_counter() - t0

    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = _extract_metrics(list(trainer.state.log_history))
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    mem_alloc = None
    if torch.cuda.is_available():
        mem_alloc = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

    summary = {
        "status": "completed",
        "run_name": run_dir.name,
        "model_id": model_id,
        "method": method,
        "use_4bit": use_4bit,
        "adapter_path": str(adapter_dir.relative_to(project_root())).replace("\\", "/"),
        "train_path": str(train_file.relative_to(project_root())).replace("\\", "/")
        if train_file.is_relative_to(project_root())
        else str(train_file),
        "validation_path": str(val_file.relative_to(project_root())).replace("\\", "/")
        if val_file.is_relative_to(project_root())
        else str(val_file),
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "num_epochs_configured": float(training.get("num_epochs", 3)),
        "early_stopping_patience": patience,
        "seed": int(model_cfg.get("reproducibility", {}).get("seed", 42)),
        "lora": model_cfg.get("lora", {}),
        "quantization": model_cfg.get("quantization", {}),
        "gpu_name": gpu_name,
        "max_memory_allocated_gb": mem_alloc,
        "wall_clock_seconds": round(elapsed, 2),
        "train_loss": getattr(train_result, "training_loss", None),
        "metrics_from_trainer": {
            "final_train_loss": metrics["final_train_loss"],
            "best_eval_loss": metrics["best_eval_loss"],
            "last_eval_loss": metrics["last_eval_loss"],
            "train_runtime_seconds": metrics["train_runtime_seconds"],
            "eval_loss_by_epoch": metrics["eval_loss_by_epoch"],
        },
        "completed_at_utc": _utc_now(),
        "academic_note": (
            "Metrics below are from this run's Trainer logs only. "
            "Do not invent or backfill numbers."
        ),
    }
    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Persist full log history separately for thesis appendixices.
    (run_dir / "trainer_log_history.json").write_text(
        json.dumps(metrics["log_history"], indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Adapter saved to %s", adapter_dir)
    return summary
