# src/rag_engine.py

import sys, os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 0. Ensure project root is in sys.path
# ---------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "rag_docs"   # <— your knowledge base folder

# ---------------------------------------------------------
# 2. Load and chunk Markdown documents
# ---------------------------------------------------------

def load_documents():
    docs = []

    # Load ALL .md files inside rag_docs/
    for file in DOCS_DIR.glob("*.md"):
        text = file.read_text(encoding="utf-8")

        # Split into chunks by blank lines
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

        for chunk in chunks:
            docs.append((file.name, chunk))

    return docs

# ---------------------------------------------------------
# 3. Embedding model
# ---------------------------------------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

DOCUMENTS = load_documents()
EMBEDDINGS = model.encode([d[1] for d in DOCUMENTS])  # embed only text

# ---------------------------------------------------------
# 4. Retrieve top‑k relevant chunks
# ---------------------------------------------------------

def retrieve_context(query: str, k=3):
    q_emb = model.encode([query])[0]
    scores = np.dot(EMBEDDINGS, q_emb)
    top_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for idx in top_idx:
        name, text = DOCUMENTS[idx]
        results.append(f"[{name}]\n{text}\n")

    return "\n".join(results)

# ---------------------------------------------------------
# 5. Import check (optional)
# ---------------------------------------------------------

try:
    import chromadb
    import openai
    print("✓ All imports successful")
except Exception as e:
    print(f"Import error: {e}")
