
import streamlit as st
import tempfile
import faiss
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import resampy
from pathlib import Path

INDEX_PATH = Path("artifacts/faiss_yamnet.index")
MAPPING_PATH = Path("artifacts/id_to_path.txt")

yamnet_model_handle = "/home/student/MIR_project/venv/models"
yamnet_model = hub.load(yamnet_model_handle)

def load_mapping():
    id2path = {}
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            idx, path = line.strip().split("\t")
            id2path[int(idx)] = path
    return id2path

def load_faiss_index():
    return faiss.read_index(str(INDEX_PATH))

def load_audio_resample(file_path, target_sr=16000):
    audio, sr = sf.read(str(file_path))
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio

def embed_audio(audio_path):
    audio = load_audio_resample(audio_path)
    scores, embeddings, spectrogram = yamnet_model(audio)
    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    emb_mean = emb_mean / np.linalg.norm(emb_mean)
    return emb_mean.astype("float32")

st.title("YAMNet Speech/Audio Retrieval Demo")

uploaded_file = st.file_uploader("Upload your audio query (wav, mp3, ogg)", type=["wav", "mp3", "ogg"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    st.audio(tmp_path)

    index = load_faiss_index()
    id2path = load_mapping()

    q_emb = embed_audio(tmp_path)
    q_emb = np.expand_dims(q_emb, axis=0)
    scores, ids = index.search(q_emb, 10)

    st.subheader("Top Matches")
    for i, idx in enumerate(ids[0]):
        path = id2path[idx]
        st.write(f"{i+1}. {path}  |  Score: {scores[0][i]:.4f}")
        st.audio(path)
