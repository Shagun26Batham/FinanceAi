import os
import csv
import hashlib
import logging
from pathlib import Path

import chromadb
import ollama

DATA_DIR = "data"
CHROMA_PATH = "db/chroma_db"
COLLECTION_NAME = "rag_chatbot"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 75
BATCH_SIZE = 32

logging.basicConfig(level=logging.INFO, format="[INGEST] %(message)s")
logger = logging.getLogger(__name__)


def load_all_csvs(data_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        logger.error(f"Data directory '{data_dir}' does not exist.")
        return []

    rows = []
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in '{data_dir}'.")
        return []

    logger.info(f"CSV files found: {len(csv_files)}")

    for csv_path in csv_files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parts = []
                for k, v in row.items():
                    if not k or not v:
                        continue
                    k_str = str(k).strip()
                    v_str = str(v).strip()
                    if not v_str:
                        continue
                        
                    if k_str.lower() == "text":
                        parts.append(v_str)
                    elif k_str.lower() == "question":
                        parts.append(f"Q: {v_str}")
                    elif k_str.lower() == "answer":
                        parts.append(f"A: {v_str}")
                    else:
                        parts.append(f"{k_str}: {v_str}")
                
                if parts:
                    row_text = "\n".join(parts)
                    rows.append({
                        "text": f"[SOURCE: {csv_path.name}]\n{row_text}"
                    })

    return rows


def chunk_text(text):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def prepare_chunks(rows):
    chunks = []

    for row in rows:
        for chunk in chunk_text(row["text"]):
            chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
            chunks.append({
                "text": chunk,
                "hash": chunk_hash
            })

    logger.info(f"Total chunks generated: {len(chunks)}")
    return chunks


def build_index(chunks):
    os.makedirs(CHROMA_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing = collection.get(include=["metadatas"])
    existing_hashes = set()

    for meta in existing["metadatas"]:
        if meta and "hash" in meta:
            existing_hashes.add(meta["hash"])

    new_chunks = [c for c in chunks if c["hash"] not in existing_hashes]
    logger.info(f"New chunks to index: {len(new_chunks)}")

    if not new_chunks:
        return

    try:
        for i in range(0, len(new_chunks), BATCH_SIZE):
            batch = new_chunks[i:i + BATCH_SIZE]

            texts = [c["text"] for c in batch]
            embeddings = []

            for text in texts:
                res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
                embeddings.append(res["embedding"])

            ids = [c["hash"] for c in batch]
            metadatas = [{"hash": c["hash"]} for c in batch]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
    except Exception as e:
        logger.error(f"Failed to connect to Ollama ({e}). Ensure it is running and '{EMBED_MODEL}' is pulled.")


if __name__ == "__main__":
    rows = load_all_csvs(DATA_DIR)
    if not rows:
        logger.info("No data to process. Exiting.")
    else:
        chunks = prepare_chunks(rows)
        build_index(chunks)