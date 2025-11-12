# build_metadata_index.py
import pandas as pd
import numpy as np
import os
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

    # df_dev = pd.read_csv("data/clothov2/development.csv")
    # df_dev['path'] = 'data/clothov2/development/audio/' + df_dev['file_name']

    # df_eval = pd.read_csv("data/clothov2/evaluation.csv")
    # df_eval['path'] = 'data/clothov2/evaluation/audio/' + df_eval['file_name']

    # df = pd.concat([df_dev, df_eval])
    df = pd.read_csv("data/clothov2/development.csv")   # or evaluation.csv
    df['path'] = df['file_name'].apply(lambda x: os.path.join("data/clothov2/development/development", x))
   

# --- Combine multiple caption columns into a single metadata field ---
    caption_cols = [col for col in df.columns if col.startswith("caption_")]

    if caption_cols:
    # Merge all caption columns into one text field separated by " | "
        df["metadata"] = df[caption_cols].fillna("").agg(" | ".join, axis=1)
    else:
    # Fallback in case no caption columns exist
        if 'caption' in df.columns:
            df.rename(columns={'caption': 'metadata'}, inplace=True)
        elif 'text' in df.columns:
            df.rename(columns={'text': 'metadata'}, inplace=True)
        else:
            df['metadata'] = "No caption"

    
    df[['path', 'metadata']].to_csv("data/metadata.csv", index=False)

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