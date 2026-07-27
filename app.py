from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from groundedops.answering import AnswerGenerator, REFUSAL
from groundedops.config import POLICY_DIR, RETRIEVE_K, RERANK_K, UPLOAD_DIR
from groundedops.conflicts import detect_conflicts
from groundedops.documents import chunk_documents, load_documents
from groundedops.evaluation import run_evaluation
from groundedops.retrieval import GroundedRetriever


st.set_page_config(page_title="GroundedOps", page_icon=":material/article:", layout="wide")


def css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #17201b;
          --muted: #5e6a63;
          --line: #d8dfd9;
          --wash: #f6f7f2;
          --accent: #176b5b;
          --warn: #9b4d16;
        }
        .stApp { background: var(--wash); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(246, 247, 242, 0.86); }
        .block-container { padding-top: 2rem; max-width: 1280px; }
        .metric-card {
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 14px 16px;
          background: #fffefa;
          min-height: 94px;
        }
        .metric-label { color: var(--muted); font-size: 0.82rem; }
        .metric-value { color: var(--ink); font-size: 1.65rem; font-weight: 700; margin-top: 4px; }
        .citation {
          border-left: 4px solid var(--accent);
          background: #fffefa;
          padding: 12px 14px;
          margin: 10px 0;
          border-radius: 0 8px 8px 0;
        }
        .source-title { font-weight: 700; color: var(--ink); }
        .small-muted { color: var(--muted); font-size: 0.86rem; }
        .conflict {
          border: 1px solid #e0c9b6;
          border-radius: 8px;
          padding: 14px;
          background: #fff8f1;
        }
        div[data-testid="stButton"] button {
          border-radius: 8px;
          border: 1px solid var(--accent);
          color: var(--accent);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_retriever() -> GroundedRetriever:
    return GroundedRetriever()


@st.cache_resource(show_spinner=False)
def get_generator() -> AnswerGenerator:
    return AnswerGenerator()


def prepare_uploads(files) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for file in files:
        target = UPLOAD_DIR / file.name
        target.write_bytes(file.getbuffer())


def load_corpus(include_uploads: bool) -> tuple[list, list]:
    paths = list(POLICY_DIR.glob("*"))
    if include_uploads and UPLOAD_DIR.exists():
        paths += list(UPLOAD_DIR.glob("*"))
    docs = load_documents(paths)
    return docs, chunk_documents(docs)


def rebuild_index(include_uploads: bool) -> tuple[list, list]:
    retriever = get_retriever()
    docs, chunks = load_corpus(include_uploads)
    retriever.reset()
    retriever.index(chunks)
    return docs, chunks


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def render_citation(idx: int, result) -> None:
    score = result.rerank_score if result.rerank_score is not None else result.retrieval_score
    st.markdown(
        f"""
        <div class="citation">
          <div class="source-title">[{idx}] {result.document}</div>
          <div class="small-muted">Chunk {result.chunk_index} | score {score:.3f}</div>
          <div>{result.text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    css()
    retriever = get_retriever()
    generator = get_generator()

    with st.sidebar:
        st.title("GroundedOps")
        st.caption("AI policy and document assistant")
        uploaded = st.file_uploader(
            "Upload policy files",
            type=["txt", "md", "pdf", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        include_uploads = st.toggle("Include uploaded documents", value=True)
        if uploaded:
            prepare_uploads(uploaded)
            st.success(f"Saved {len(uploaded)} file(s).")
        if st.button("Rebuild document index", use_container_width=True):
            with st.spinner("Embedding and indexing documents..."):
                rebuild_index(include_uploads)
            st.success("Index rebuilt.")
        st.divider()
        st.caption("Cohere mode" if retriever.using_cohere else "Local demo mode")
        if not os.getenv("COHERE_API_KEY"):
            st.info("Set COHERE_API_KEY to use embed-v4.0, rerank-v4.0-fast, and command-a-plus-05-2026.")

    docs, chunks = load_corpus(include_uploads)
    if retriever.collection.count() == 0 and chunks:
        retriever.index(chunks)

    left, right = st.columns([0.62, 0.38], vertical_alignment="top")

    with left:
        st.title("GroundedOps")
        st.subheader("Ask policy questions with citations, refusal behavior, and retrieval evaluation.")
        query = st.text_input(
            "Question",
            value="Which policies disagree about the expense submission deadline?",
            label_visibility="collapsed",
            placeholder="Ask about expenses, access, remote work, incidents, or procurement...",
        )
        ask = st.button("Ask grounded question", type="primary")

        if ask and query:
            with st.spinner("Retrieving 20 passages, reranking, and generating a grounded answer..."):
                retrieved, retrieve_ms = retriever.retrieve(query, RETRIEVE_K)
                reranked, rerank_ms = retriever.rerank(query, retrieved, RERANK_K)
                answer = generator.answer(query, reranked)

            if answer.answer.strip() == REFUSAL:
                st.warning(answer.answer)
            else:
                st.markdown(answer.answer)

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Retrieved candidates", str(len(retrieved)))
            with c2:
                metric_card("Reranked passages", str(len(reranked)))
            with c3:
                metric_card("Search latency", f"{retrieve_ms + rerank_ms:.0f} ms")

            st.markdown("**Citations**")
            if answer.citations:
                for idx, result in enumerate(answer.citations, start=1):
                    render_citation(idx, result)
            else:
                st.caption("No supporting passage passed the evidence threshold.")

            with st.expander("Before and after rerank"):
                rows = []
                for rank, item in enumerate(retrieved[:8], start=1):
                    rows.append(
                        {
                            "Rank": rank,
                            "Before rerank": item.document,
                            "Retrieval score": round(item.retrieval_score, 3),
                            "After rerank": reranked[rank - 1].document if rank <= len(reranked) else "",
                            "Rerank score": round(reranked[rank - 1].rerank_score or 0, 3) if rank <= len(reranked) else "",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with right:
        st.markdown("**Corpus**")
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Documents", str(len(docs)))
        with c2:
            metric_card("Passages", str(len(chunks)))
        with st.expander("Indexed documents", expanded=True):
            for doc in docs:
                st.write(f"**{doc.name}**")
                st.caption(Path(doc.source_path).name)

        st.markdown("**Conflict Detector**")
        conflicts = detect_conflicts(chunks)
        if not conflicts:
            st.caption("No obvious numeric policy conflicts found.")
        for conflict in conflicts:
            st.markdown(
                f"""
                <div class="conflict">
                  <strong>{conflict.topic}</strong><br>
                  <span class="small-muted">{conflict.document_a} conflicts with {conflict.document_b}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Show conflicting passages"):
                st.write(f"**{conflict.document_a}**")
                st.write(conflict.passage_a)
                st.write(f"**{conflict.document_b}**")
                st.write(conflict.passage_b)

    st.divider()
    st.markdown("**Evaluation Dashboard**")
    if st.button("Run 18-question evaluation"):
        with st.spinner("Comparing retrieval before and after rerank..."):
            summary = run_evaluation(retriever, generator)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Before rerank", f"{summary.metrics['retrieval_accuracy_before']:.0%}")
        with m2:
            metric_card("After rerank", f"{summary.metrics['retrieval_accuracy_after']:.0%}")
        with m3:
            metric_card("Citation coverage", f"{summary.metrics['citation_coverage']:.0%}")
        with m4:
            metric_card("Avg response", f"{summary.metrics['avg_response_time_ms']:.0f} ms")
        st.dataframe(summary.rows, hide_index=True, use_container_width=True)

    if st.sidebar.button("Clear uploaded files", use_container_width=True):
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        st.sidebar.success("Uploaded files cleared.")


if __name__ == "__main__":
    main()
