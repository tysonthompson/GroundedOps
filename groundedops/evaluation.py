from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from groundedops.answering import AnswerGenerator
from groundedops.retrieval import GroundedRetriever


TEST_QUESTIONS = [
    ("How quickly must expense reports be submitted under the current expense policy?", "Expense Policy 2026"),
    ("Which document says travel expense reports can be submitted within 60 days?", "Legacy Travel Handbook"),
    ("Can employees expense alcohol at ordinary meals?", "Expense Policy 2026"),
    ("When is business class allowed?", "Expense Policy 2026"),
    ("How often are production access reviews performed?", "Security Access Policy"),
    ("Who must approve production access?", "Security Access Policy"),
    ("How soon must lost devices be reported?", "Security Access Policy"),
    ("Can employees use personal devices to store customer data?", "Remote Work Guidelines"),
    ("What are core collaboration hours?", "Remote Work Guidelines"),
    ("How many business days may an employee work from another country?", "Remote Work Guidelines"),
    ("What counts as a Severity 1 incident?", "Incident Response Playbook"),
    ("How often are internal updates required for Severity 1 incidents?", "Incident Response Playbook"),
    ("When must Severity 2 post-incident reviews be completed?", "Incident Response Playbook"),
    ("Who approves purchases under 1,000 USD?", "Procurement Policy"),
    ("What approvals are needed above 10,000 USD?", "Procurement Policy"),
    ("When is competitive vendor review required?", "Procurement Policy"),
    ("Do new tools storing customer data need security review?", "Procurement Policy"),
    ("What is the company's pet insurance reimbursement policy?", ""),
]


@dataclass
class EvalSummary:
    rows: pd.DataFrame
    metrics: dict[str, float]


def run_evaluation(retriever: GroundedRetriever, generator: AnswerGenerator) -> EvalSummary:
    rows = []
    for question, expected_doc in TEST_QUESTIONS:
        retrieval_results, retrieve_ms = retriever.retrieve(question)
        reranked, rerank_ms = retriever.rerank(question, retrieval_results)
        started = time.perf_counter()
        answer = generator.answer(question, reranked)
        answer_ms = (time.perf_counter() - started) * 1000
        before_hit = bool(expected_doc and retrieval_results and retrieval_results[0].document == expected_doc)
        after_hit = bool(expected_doc and reranked and reranked[0].document == expected_doc)
        unsupported_ok = not expected_doc and not answer.supported
        rows.append(
            {
                "Question": question,
                "Expected document": expected_doc or "Unsupported",
                "Top before rerank": retrieval_results[0].document if retrieval_results else "-",
                "Top after rerank": reranked[0].document if reranked else "-",
                "Before correct": before_hit,
                "After correct": after_hit,
                "Citation coverage": bool(answer.citations) if expected_doc else unsupported_ok,
                "Response time ms": round(retrieve_ms + rerank_ms + answer_ms, 1),
            }
        )
    frame = pd.DataFrame(rows)
    supported = frame[frame["Expected document"] != "Unsupported"]
    metrics = {
        "retrieval_accuracy_before": float(supported["Before correct"].mean()) if not supported.empty else 0.0,
        "retrieval_accuracy_after": float(supported["After correct"].mean()) if not supported.empty else 0.0,
        "citation_coverage": float(frame["Citation coverage"].mean()) if not frame.empty else 0.0,
        "avg_response_time_ms": float(frame["Response time ms"].mean()) if not frame.empty else 0.0,
    }
    return EvalSummary(rows=frame, metrics=metrics)
