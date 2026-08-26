"""Chat-template formatting for Dataset B instruction rows."""

from __future__ import annotations

from typing import Any


def build_user_content(instruction: str, input_text: str = "") -> str:
    """Combine instruction and optional input into a single user message."""
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a Dataset B row into chat messages for Qwen-style instruct models."""
    return [
        {
            "role": "user",
            "content": build_user_content(
                str(row.get("instruction", "")),
                str(row.get("input", "")),
            ),
        },
        {"role": "assistant", "content": str(row.get("response", "")).strip()},
    ]


def messages_to_text_fallback(messages: list[dict[str, str]]) -> str:
    """Qwen ChatML-style fallback used for dry-runs without downloading a tokenizer."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def row_to_text(row: dict[str, Any], tokenizer: Any | None = None) -> str:
    """Format one FT example as a single supervised training string."""
    messages = row_to_messages(row)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return messages_to_text_fallback(messages)
