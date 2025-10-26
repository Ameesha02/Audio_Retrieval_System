# app_fusion.py
import streamlit as st
import tempfile
import numpy as np
import faiss
import resampy
import soundfile as sf
import tensorflow_hub as hub
import tensorflow as tf
from pathlib import Path
from sentence_transformers import SentenceTransformer
import whisper

from utils_paths import (
    AUDIO_INDEX_PATH, AUDIO_MAPPING_PATH,
    META_INDEX_PATH, META_MAPPING_PATH
)

st.set_page_config(page_title="Audio + Metadata Fusion Retrieval", layout="centered")

# Models
@st.cache_resource
def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")

@st.cache_resource
def load_text_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_asr():
    import whisper
    return whisper.load_model("base")

@st.cache_resource
def load_index_and_map(idx_path, map_path):
    index = faiss.read_index(str(idx_path))
    id2path = {}
    with open(map_path, "r", encoding="utf-8") as f:
        for line in f:
            idx, p = line.strip().split("\t", 1)
            id2path[int(idx)] = p
    return index, id2path

yamnet = load_yamnet()
text_encoder = load_text_encoder()
asr = load_asr()

a_index, a_id2path = load_index_and_map(AUDIO_INDEX_PATH, AUDIO_MAPPING_PATH)
m_index, m_id2path = load_index_and_map(META_INDEX_PATH, META_MAPPING_PATH)

def load_audio_resample(file_path, target_sr=16000):
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

def minmax_norm(arr):
    if arr.size == 0:
        return arr
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 1e-12:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

st.title("Audio Retrieval with Late Fusion (Audio + Metadata)")

uploaded = st.file_uploader("Upload a speech query (wav/mp3/ogg)", type=["wav", "mp3", "ogg"])
wa = st.slider("Audio weight (wa)", 0.0, 1.0, 0.5, 0.05)
wm = 1.0 - wa
topk = st.slider("Top-K", 3, 20, 10, 1)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        qpath = Path(tmp.name)
    st.audio(str(qpath))

    # Transcribe
    with st.spinner("Transcribing..."):
        asr_res = asr.transcribe(str(qpath))
        qtext = asr_res.get("text", "").strip()
    st.write(f"Transcript: {qtext}")

    # Audio search
    q_a = yamnet_embed(qpath)[None, :]
    a_scores, a_ids = a_index.search(q_a, topk)
    a_scores = a_scores[0]
    a_ids = a_ids[0]

    # Metadata search
    q_m = text_encoder.encode([qtext], normalize_embeddings=True).astype("float32")
    m_scores, m_ids = m_index.search(q_m, topk)
    m_scores = m_scores[0]
    m_ids = m_ids[0]

    # Normalize
    a_norm = minmax_norm(a_scores)
    m_norm = minmax_norm(m_scores)
    a_map = {int(i): float(s) for i, s in zip(a_ids, a_norm)}
    m_map = {int(i): float(s) for i, s in zip(m_ids, m_norm)}

    # Union and fuse
    cand = list(set(a_ids.tolist()) | set(m_ids.tolist()))
    rows = []
    for cid in cand:
        sa = a_map.get(int(cid), 0.0)
        sm = m_map.get(int(cid), 0.0)
        fused = wa * sa + wm * sm
        path = m_id2path.get(int(cid), a_id2path.get(int(cid), ""))
        rows.append((fused, cid, path, sa, sm))
    rows.sort(key=lambda x: x[0], reverse=True)

    st.subheader("Results (fused)")
    for rank, (fused, cid, pth, sa, sm) in enumerate(rows[:topk], 1):
        st.write(f"{rank}. score={fused:.3f} | audio={sa:.3f} | meta={sm:.3f} | {pth}")
        st.audio(pth)