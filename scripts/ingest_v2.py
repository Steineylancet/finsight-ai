"""
FinSight 2.0 — Ingest v2
Reads all markdown docs from corpus/, chunks by section,
embeds with ada-002, uploads to a fresh Azure AI Search index.

Run AFTER generate_corpus.py:
    python scripts/ingest_v2.py
"""

import os, sys, re, uuid, logging
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.azure_openai_client import AzureOpenAIClient
from backend.azure_search import AzureSearchClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")
MAX_TOKENS = 500

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from a markdown file."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta, parts[2].strip()


def chunk_document(meta: dict, body: str) -> list[dict]:
    """
    Split a document into chunks by ## section headers.
    If a section is too long, split further by paragraph.
    Each chunk carries the document metadata.
    """
    title  = meta.get("title", "Unknown Document")
    dept   = meta.get("department", "")
    dtype  = meta.get("doc_type", "")
    fy     = meta.get("fiscal_year", "")
    qtr    = meta.get("quarter", "")

    # Split on ## headings (keep the heading with its content)
    sections = re.split(r"\n(?=## )", body)
    chunks   = []

    for section in sections:
        section = section.strip()
        if not section or count_tokens(section) < 30:
            continue

        if count_tokens(section) <= MAX_TOKENS:
            chunks.append(_make_chunk(section, title, dtype, dept, fy, qtr, len(chunks)))
        else:
            # Split long section by paragraph
            paras = [p.strip() for p in section.split("\n\n") if p.strip()]
            current, current_tokens = [], 0
            for para in paras:
                pt = count_tokens(para)
                if current_tokens + pt > MAX_TOKENS and current:
                    chunks.append(_make_chunk(
                        "\n\n".join(current), title, dtype, dept, fy, qtr, len(chunks)
                    ))
                    current, current_tokens = [], 0
                current.append(para)
                current_tokens += pt
            if current:
                chunks.append(_make_chunk(
                    "\n\n".join(current), title, dtype, dept, fy, qtr, len(chunks)
                ))

    return chunks


def _make_chunk(content, title, dtype, dept, fy, qtr, idx):
    return {
        "id":               str(uuid.uuid4()),
        "content":          content,
        "title":            title,
        "data_type":        dtype,
        "entity":           "Crestwood Capital Group",
        "department":       dept,
        "fiscal_year":      fy,
        "fiscal_period":    qtr,
        "expense_category": "",
        "chunk_index":      idx,
    }


def read_corpus(corpus_dir: str) -> list[dict]:
    """Walk corpus/ and return all chunks from all markdown files."""
    all_chunks = []
    n_docs = 0

    for root, _, files in os.walk(corpus_dir):
        for fname in sorted(files):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, encoding="utf-8") as f:
                text = f.read()

            meta, body = parse_frontmatter(text)
            if not body.strip():
                continue

            chunks = chunk_document(meta, body)
            all_chunks.extend(chunks)
            n_docs += 1

    logger.info(f"Read {n_docs} documents → {len(all_chunks)} chunks")
    return all_chunks


def main():
    logger.info("=" * 65)
    logger.info("FinSight 2.0 — Ingest v2")
    logger.info("=" * 65)

    openai_client  = AzureOpenAIClient()
    search_client  = AzureSearchClient()

    # ── Step 1: Recreate index ────────────────────────────────────────────────
    logger.info("\n[Step 1] Recreating Azure AI Search index...")
    search_client.delete_index()
    search_client.create_index()

    # ── Step 2: Read and chunk corpus ─────────────────────────────────────────
    logger.info("\n[Step 2] Reading corpus documents...")
    all_chunks = read_corpus(CORPUS_DIR)

    if not all_chunks:
        logger.error("No chunks found. Run generate_corpus.py first.")
        return

    # Chunk type breakdown
    from collections import Counter
    type_counts = Counter(c["data_type"] for c in all_chunks)
    for dtype, count in sorted(type_counts.items()):
        logger.info(f"  {dtype}: {count} chunks")

    if len(all_chunks) > 9_800:
        raise RuntimeError(
            f"Chunk count {len(all_chunks)} exceeds free-tier limit (9,800). "
            "Reduce corpus size before proceeding."
        )

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    logger.info(f"\n[Step 3] Generating embeddings for {len(all_chunks)} chunks...")
    texts      = [c["content"] for c in all_chunks]
    embeddings = openai_client.get_embeddings_batch(texts, batch_size=16)
    for i, chunk in enumerate(all_chunks):
        chunk["embedding"] = embeddings[i]

    # ── Step 4: Upload ────────────────────────────────────────────────────────
    logger.info(f"\n[Step 4] Uploading to Azure AI Search...")
    search_client.upload_documents(all_chunks, batch_size=100)

    logger.info("\n" + "=" * 65)
    logger.info("Ingest complete!")
    logger.info(f"  Total chunks indexed: {len(all_chunks)}")
    logger.info("\nNext: python -m uvicorn backend.main:app --port 8000")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
