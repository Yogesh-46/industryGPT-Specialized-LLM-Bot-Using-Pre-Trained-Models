"""Bounded session-level conversation memory (in-process only)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ConversationMemory:
    """Keep a short rolling chat history for follow-up questions."""

    max_turns: int = 8
    max_history_chars: int = 6000
    turns: list[Turn] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        self._trim()

    def clear(self) -> None:
        self.turns.clear()

    def as_chat_messages(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def _trim(self) -> None:
        # Keep last N user+assistant messages (approx turns*2)
        max_messages = max(2, self.max_turns * 2)
        if len(self.turns) > max_messages:
            self.turns = self.turns[-max_messages:]

        total = sum(len(t.content) for t in self.turns)
        while total > self.max_history_chars and len(self.turns) > 2:
            removed = self.turns.pop(0)
            total -= len(removed.content)
