import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import soundfile as sf
import resampy

from utils_paths import (
    AUDIO_INDEX_PATH, AUDIO_MAPPING_PATH,
    META_INDEX_PATH, META_MAPPING_PATH
)


def load_index_and_map(index_path, map_path):
    index = faiss.read_index(str(index_path))
    id2path = {}

    with open(map_path, "r") as f:
        for line in f:
            i, p = line.strip().split("\t")
            id2path[int(i)] = p

    return index, id2path



def compute_relevance(gt_path, ranked_ids, id2path):
    """Return binary relevance list: 1 if file matches GT."""
    rel = []
    for rid in ranked_ids:
        rel.append(1 if id2path[rid].endswith(gt_path) else 0)
    return rel


def average_precision(rel, k=10):
    rel = rel[:k]
    if sum(rel) == 0:
        return 0.0

    score = 0.0
    hits = 0

    for i, r in enumerate(rel, start=1):
        if r == 1:
            hits += 1
            score += hits / i   # precision@i each time we find relevant

    return score / hits


def ndcg(rel, k=10):
    rel = np.array(rel[:k])
    dcg = np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2)))

    ideal = np.sort(rel)[::-1]
    idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))

    return 0.0 if idcg == 0 else dcg / idcg



import tensorflow_hub as hub
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
    return audio

def yamnet_embed(path):
    audio = load_audio(path)
    _, embeddings, _ = yamnet(audio)
    emb = tf.reduce_mean(embeddings, axis=0).numpy().astype("float32")
    emb /= (np.linalg.norm(emb) + 1e-12)
    return emb.reshape(1, -1)



def evaluate_query(
    query_text, gt_path,
    t_encoder,
    a_index, a_id2path,
    m_index, m_id2path,
    wa=0.40, topk=10,
):


    gt_id = None
    for i, p in m_id2path.items():
        if p.endswith(gt_path):
            gt_id = i
            break
    if gt_id is None:
        return None


    q_text_emb = t_encoder.encode([query_text], normalize_embeddings=True).astype('float32')
    m_scores, m_ids = m_index.search(q_text_emb, topk)
    m_scores = m_scores[0]
    m_ids = m_ids[0]

    gt_full_path = None
    for p in a_id2path.values():
        if p.endswith(gt_path):
            gt_full_path = p
            break

    if gt_full_path is None:
        return None

    q_audio_emb = yamnet_embed(gt_full_path)
    a_scores, a_ids = a_index.search(q_audio_emb, topk)
    a_scores = a_scores[0]
    a_ids = a_ids[0]

    # ------------------------------------
    # Normalize scores (0-1)
    # ------------------------------------
    def minmax(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.ones_like(x) * 0.5
        return (x - mn) / (mx - mn)

    a_norm = minmax(a_scores)
    m_norm = minmax(m_scores)

    # ------------------------------------
    # Fusion (late fusion)
    # ------------------------------------
    candidates = list(set(a_ids.tolist()) | set(m_ids.tolist()))
    fused_scores = {}

    for cid in candidates:
        sa = a_norm[list(a_ids).index(cid)] if cid in a_ids else 0
        sm = m_norm[list(m_ids).index(cid)] if cid in m_ids else 0
        fused_scores[cid] = wa * sa + (1 - wa) * sm

    fused_sorted = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    fused_ids = [cid for cid, _ in fused_sorted[:topk]]

    rel_M = compute_relevance(gt_path, m_ids, m_id2path)
    rel_A = compute_relevance(gt_path, a_ids, a_id2path)
    rel_F = compute_relevance(gt_path, fused_ids, a_id2path)


    return {
        
        "R5_M": max(rel_M[:5]),
        "R10_M": max(rel_M[:10]),
        "AP_M": average_precision(rel_M, 10),

        
        "R5_A": max(rel_A[:5]),
        "R10_A": max(rel_A[:10]),
        "AP_A": average_precision(rel_A, 10),

        
        "R5_F": max(rel_F[:5]),
        "R10_F": max(rel_F[:10]),
        "AP_F": average_precision(rel_F, 10),
        "nDCG_F": ndcg(rel_F, 10),
    }


# ===============================================
# MAIN
# ===============================================
def main():

    # Load indexes
    a_index, a_id2path = load_index_and_map(AUDIO_INDEX_PATH, AUDIO_MAPPING_PATH)
    m_index, m_id2path = load_index_and_map(META_INDEX_PATH, META_MAPPING_PATH)

    # Text encoder
    t_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Evaluation CSV
    df = pd.read_csv("data/clothov2/evaluation.csv")

    results = []

    for row in tqdm(df.itertuples(), total=len(df)):
        caption = row.caption_1
        filename = row.file_name

        out = evaluate_query(
            caption,
            filename,
            t_encoder,
            a_index, a_id2path,
            m_index, m_id2path,
            wa=0.40,
            topk=10
        )

        if out:
            results.append(out)

    results = pd.DataFrame(results)
    results.to_csv("fusion_evaluation_results.csv", index=False)

    print("RESULTS COLUMNS:", results.columns.tolist())

    def pct(x):
        return round(float(x) * 100, 2)



    # ---- METADATA ----
    print("---- Metadata Retrieval ----")
    print("mAP@10:", pct(results["AP_M"].mean()))
    print("R@5   :", pct(results["R5_M"].mean()))
    print("R@10  :", pct(results["R10_M"].mean()))
    print()

    # ---- AUDIO ----
    print("---- Audio Retrieval ----")
    print("mAP@10:", pct(results["AP_A"].mean()))
    print("R@5   :", pct(results["R5_A"].mean()))
    print("R@10  :", pct(results["R10_A"].mean()))
    print()

    # ---- FUSION ----
    print("---- Fusion Retrieval ----")
    print("mAP@10:", pct(results["AP_F"].mean()))
    print("nDCG   :", pct(results["nDCG_F"].mean()))
    print("R@5    :", pct(results["R5_F"].mean()))
    print("R@10   :", pct(results["R10_F"].mean()))
    print("\n===============================================\n")


if __name__ == "__main__":
    main()
