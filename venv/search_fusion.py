# search_fusion.py
import numpy as np
import faiss
import soundfile as sf
import resampy
import tensorflow_hub as hub
import tensorflow as tf
from pathlib import Path
from sentence_transformers import SentenceTransformer
import whisper

from utils_paths import (
    AUDIO_INDEX_PATH, AUDIO_MAPPING_PATH,
    META_INDEX_PATH, META_MAPPING_PATH
)

# YAMNet model (audio space)
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
yamnet = hub.load(YAMNET_HANDLE)

# Metadata encoder (text space)
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
text_encoder = SentenceTransformer(TEXT_MODEL)

def read_mapping(path: Path):
    id2path = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            idx, p = line.strip().split("\t", 1)
            id2path[int(idx)] = p
    return id2path

def load_audio_resample(file_path: Path, target_sr=16000):
    audio, sr = sf.read(str(file_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
    return audio

def yamnet_embed(file_path: Path) -> np.ndarray:
    audio = load_audio_resample(file_path)
    scores, embeddings, _ = yamnet(audio)
    emb = tf.reduce_mean(embeddings, axis=0).numpy().astype("float32")
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb

def minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 1e-12:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query_audio", type=str, help="Path to query audio (wav/mp3/ogg)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--wa", type=float, default=0.5, help="Weight for audio similarity")
    ap.add_argument("--wm", type=float, default=0.5, help="Weight for metadata similarity")
    ap.add_argument("--print_transcript", action="store_true")
    args = ap.parse_args()

    # Load indices and mappings
    a_index = faiss.read_index(str(AUDIO_INDEX_PATH))
    a_id2path = read_mapping(AUDIO_MAPPING_PATH)

    m_index = faiss.read_index(str(META_INDEX_PATH))
    m_id2path = read_mapping(META_MAPPING_PATH)

    # Audio branch: embed query audio with YAMNet and search
    q_a = yamnet_embed(Path(args.query_audio))[None, :]
    a_scores, a_ids = a_index.search(q_a, args.topk)
    a_scores = a_scores[0]
    a_ids = a_ids[0]

    # Metadata branch: transcribe → embed text → search metadata index
    asr = whisper.load_model("base")
    asr_res = asr.transcribe(args.query_audio)
    query_text = asr_res.get("text", "").strip()
    if args.print_transcript:
        print(f"Transcript: {query_text}")

    q_m = text_encoder.encode([query_text], normalize_embeddings=True).astype("float32")
    m_scores, m_ids = m_index.search(q_m, args.topk)
    m_scores = m_scores[0]
    m_ids = m_ids[0]

    # Merge candidates: union of ids from both lists
    candidate_ids = set(a_ids.tolist()) | set(m_ids.tolist())
    candidate_ids = list(candidate_ids)

    # Build score dicts for quick lookup; normalize to [0,1]
    a_norm = minmax_norm(a_scores)
    m_norm = minmax_norm(m_scores)
    a_score_map = {int(i): float(s) for i, s in zip(a_ids, a_norm)}
    m_score_map = {int(i): float(s) for i, s in zip(m_ids, m_norm)}

    # Weighted fusion
    results = []
    for cid in candidate_ids:
        sa = a_score_map.get(int(cid), 0.0)
        sm = m_score_map.get(int(cid), 0.0)
        fused = args.wa * sa + args.wm * sm
        # Use metadata mapping for path; both mappings must align same order of items
        path = m_id2path.get(int(cid), a_id2path.get(int(cid), ""))
        results.append((fused, int(cid), path, sa, sm))

    # Sort by fused score
    results.sort(key=lambda x: x[0], reverse=True)
    print("Rank | Fused  | Audio  | Meta   | Path")
    print("-"*80)
    for rnk, (fused, cid, path, sa, sm) in enumerate(results[:args.topk], 1):
        print(f"{rnk:>4} | {fused:>6.3f} | {sa:>6.3f} | {sm:>6.3f} | {path}")

if __name__ == "__main__":
    main()