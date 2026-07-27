from __future__ import annotations

import re
from dataclasses import dataclass

from groundedops.documents import Chunk


@dataclass
class Conflict:
    topic: str
    document_a: str
    passage_a: str
    document_b: str
    passage_b: str


def detect_conflicts(chunks: list[Chunk]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    expense_windows = []
    for chunk in chunks:
        text = chunk.text.lower()
        if "expense" not in text or "submit" not in text:
            continue
        for days in re.findall(r"(\d+)\s+calendar days", text):
            expense_windows.append((int(days), chunk))
    for i, (days_a, chunk_a) in enumerate(expense_windows):
        for days_b, chunk_b in expense_windows[i + 1 :]:
            if days_a != days_b and chunk_a.document != chunk_b.document:
                conflicts.append(
                    Conflict(
                        topic="Expense submission deadline",
                        document_a=chunk_a.document,
                        passage_a=chunk_a.text,
                        document_b=chunk_b.document,
                        passage_b=chunk_b.text,
                    )
                )
    return conflicts
