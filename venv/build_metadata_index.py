# build_metadata_index.py
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils_paths import (
    PROJECT_ROOT, ARTIFACTS_DIR, META_CSV_PATH,
    META_INDEX_PATH, META_MAPPING_PATH
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not META_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing metadata CSV at {META_CSV_PATH}. Create it or run the auto-generator.")

    df = pd.read_csv(META_CSV_PATH)
    if "path" not in df.columns or "metadata" not in df.columns:
        raise ValueError("metadata.csv must contain columns: path, metadata")

    model = SentenceTransformer(EMBED_MODEL)
    texts = df["metadata"].fillna("").astype(str).tolist()
    paths = df["path"].astype(str).tolist()

    print(f"Encoding {len(texts)} metadata texts with {EMBED_MODEL} ...")
    X = model.encode(texts, normalize_embeddings=True)  # shape (N, 384)
    X = X.astype("float32")

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, str(META_INDEX_PATH))
    with open(META_MAPPING_PATH, "w", encoding="utf-8") as f:
        for i, p in enumerate(paths):
            f.write(f"{i}\t{p}\n")

    print(f"Saved metadata index to {META_INDEX_PATH}")
    print(f"Saved mapping to {META_MAPPING_PATH}")

if __name__ == "__main__":
    main()