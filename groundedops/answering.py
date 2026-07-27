from __future__ import annotations

import os
import re
from dataclasses import dataclass

import cohere

from groundedops.config import CHAT_MODEL
from groundedops.retrieval import SearchResult


@dataclass
class GroundedAnswer:
    answer: str
    supported: bool
    citations: list[SearchResult]


REFUSAL = "I couldn't find this in the provided documents."


class AnswerGenerator:
    def __init__(self) -> None:
        api_key = os.getenv("COHERE_API_KEY")
        self.co = cohere.ClientV2(api_key=api_key) if api_key else None

    def answer(self, question: str, passages: list[SearchResult]) -> GroundedAnswer:
        usable = [p for p in passages if (p.rerank_score or p.retrieval_score) > 0.16]
        if not usable:
            return GroundedAnswer(answer=REFUSAL, supported=False, citations=[])
        if self.co:
            return self._cohere_answer(question, usable)
        return self._local_answer(question, usable)

    def _cohere_answer(self, question: str, passages: list[SearchResult]) -> GroundedAnswer:
        context = "\n\n".join(
            f"[{idx}] Document: {p.document}\nPassage: {p.text}" for idx, p in enumerate(passages, start=1)
        )
        prompt = f"""
You are GroundedOps, an enterprise policy assistant. Answer only from the provided passages.
If the passages do not contain the answer, reply exactly: "{REFUSAL}"
Include bracketed citation numbers for every factual claim.

Question: {question}

Passages:
{context}
""".strip()
        response = self.co.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = response.message.content[0].text.strip()
        cited_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        cited = [p for idx, p in enumerate(passages, start=1) if idx in cited_numbers] or passages[:2]
        return GroundedAnswer(answer=text, supported=REFUSAL not in text, citations=cited)

    def _local_answer(self, question: str, passages: list[SearchResult]) -> GroundedAnswer:
        top = passages[:2]
        lines = []
        for idx, passage in enumerate(top, start=1):
            snippet = passage.text
            if len(snippet) > 420:
                snippet = snippet[:417].rstrip() + "..."
            lines.append(f"[{idx}] {snippet}")
        answer = "Based on the strongest retrieved policy passages:\n\n" + "\n\n".join(lines)
        return GroundedAnswer(answer=answer, supported=True, citations=top)
