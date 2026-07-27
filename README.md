<img width="1919" height="933" alt="groundedops" src="https://github.com/user-attachments/assets/a1fe727c-e0d8-489e-b0cf-da413cc07951" />

# GroundedOps

GroundedOps is an AI policy and document assistant built to show the engineering behind reliable enterprise RAG: document upload, semantic retrieval, Cohere Rerank, grounded answers with citations, unsupported-answer refusal, conflict detection, and an evaluation dashboard.

It is intentionally more than "chat with your PDF." The app exposes the retrieval candidates before and after reranking, shows which passages support an answer, and measures whether reranking improved top-document accuracy.

## Stack

- Frontend: Streamlit
- AI: Cohere `embed-v4.0`, `rerank-v4.0-fast`, and `command-a-plus-05-2026`
- Vector store: ChromaDB
- Document handling: text, markdown, PDF, and image uploads
- Demo data: synthetic workplace policies in `data/policies`

The app also runs in local demo mode without a Cohere key. Local mode uses deterministic embeddings and extractive answers so reviewers can open the project immediately. Set `COHERE_API_KEY` for the intended Cohere workflow.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

Set your Cohere key before running the app:

```powershell
$env:COHERE_API_KEY="..."
streamlit run app.py
```

## Architecture

```text
Uploaded files + demo policies
        |
        v
Text extraction and chunking
        |
        v
Cohere Embed v4.0 -> ChromaDB local vector index
        |
        v
Retrieve approximately 20 candidate passages
        |
        v
Cohere Rerank v4.0 fast -> top evidence passages
        |
        v
Command A Plus grounded answer with citations
        |
        v
Answer, cited passages, refusal when evidence is missing
```

## What The Demo Shows

- Reranking: retrieves broad candidates first, then reranks the top 20 passages before generation.
- Refusal behavior: returns `I couldn't find this in the provided documents.` when no passage clears the evidence threshold.
- Citations: displays the document name, chunk number, score, and source passage beside the answer.
- Conflict detection: flags conflicting expense-submission windows across the current expense policy and legacy travel handbook.
- Evaluation: runs 18 synthetic questions and reports retrieval accuracy before reranking, after reranking, citation coverage, and response time.

## Production Improvements

- Replace the Streamlit shell with a FastAPI backend and Next.js frontend for role-aware enterprise workflows.
- Add authenticated document workspaces with per-user and per-team access controls.
- Use robust OCR and layout parsing for scanned PDFs, tables, and screenshots.
- Add human review queues for detected policy conflicts.
- Persist evaluation runs and track regressions in retrieval accuracy, latency, refusal precision, and citation coverage.
- Add document versioning so stale policies can be retained for audit but excluded from current answers.

## Portfolio Response Template

Replace the bracketed sections only after the repository and live demo exist:

> I built **GroundedOps**, an AI policy and document assistant that answers questions from company documents with citations and refuses unsupported answers. It uses Cohere Embed for semantic search, ChromaDB for local vector storage, Cohere Rerank to improve passage selection, and Command A Plus for grounded responses.
>
> The project includes synthetic workplace policies, a conflict detector for contradictory policy instructions, and an evaluation dashboard comparing retrieval quality before and after reranking.
>
> Repo: [your GitHub link]  
> Demo: [your deployed app link]  
> Background: [your relevant background]
