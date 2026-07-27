from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from pypdf import PdfReader


@dataclass(frozen=True)
class Document:
    name: str
    text: str
    source_path: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    document: str
    source_path: str
    chunk_index: int


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        try:
            import pytesseract  # type: ignore

            return pytesseract.image_to_string(Image.open(path))
        except Exception:
            return (
                "Image OCR is unavailable in this environment. Install pytesseract "
                "and the Tesseract binary to extract text from screenshots."
            )
    return ""


def load_documents(paths: Iterable[Path]) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        text = normalize_text(extract_text(path))
        if text:
            docs.append(Document(name=path.stem.replace("-", " ").title(), text=text, source_path=str(path)))
    return docs


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_documents(documents: Iterable[Document], target_words: int = 115, overlap_words: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
        rolling: list[str] = []
        chunk_index = 0
        for paragraph in paragraphs:
            words = " ".join(rolling + [paragraph]).split()
            if len(words) <= target_words:
                rolling.append(paragraph)
                continue
            if rolling:
                chunk_text = " ".join(rolling)
                chunks.append(
                    Chunk(
                        id=f"{slug(doc.name)}-{chunk_index}",
                        text=chunk_text,
                        document=doc.name,
                        source_path=doc.source_path,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                rolling = [" ".join(chunk_text.split()[-overlap_words:]), paragraph]
            else:
                rolling = [paragraph]
        if rolling:
            chunks.append(
                Chunk(
                    id=f"{slug(doc.name)}-{chunk_index}",
                    text=" ".join(rolling),
                    document=doc.name,
                    source_path=doc.source_path,
                    chunk_index=chunk_index,
                )
            )
    return chunks


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
