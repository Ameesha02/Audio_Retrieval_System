
import faiss
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import resampy
from pathlib import Path
from transcribe import load_whisper, transcribe_audio
import shutil
import sys

INDEX_PATH = Path("artifacts/faiss_yamnet.index")
MAPPING_PATH = Path("artifacts/id_to_path.txt")
EMBEDDING_SIZE = 1024

yamnet_model_handle = "/home/student/MIR_project/venv/models"
yamnet_model = hub.load(yamnet_model_handle)

if shutil.which("ffmpeg") is None:
    print("Error: 'ffmpeg' not found on PATH.")
    print("Install it on Debian/Ubuntu: sudo apt update && sudo apt install -y ffmpeg")
    print("Or on Fedora: sudo dnf install ffmpeg")
    print("Or on macOS (Homebrew): brew install ffmpeg")
    sys.exit(1)

def load_audio_resample(file_path, target_sr=16000):
    audio, sr = sf.read(str(file_path))
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio

def embed_query_audio(file_path):
    audio = load_audio_resample(file_path)
    scores, embeddings, spectrogram = yamnet_model(audio)
    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    emb_mean = emb_mean / np.linalg.norm(emb_mean)
    return emb_mean.astype("float32")

def load_mapping():
    id2path = {}
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            idx, path = line.strip().split("\t")
            id2path[int(idx)] = path
    return id2path

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python search_speech_yamnet.py path/to/query_audio.wav")
        return

    query_audio_path = Path(sys.argv[1])
    index = faiss.read_index(str(INDEX_PATH))
    id2path = load_mapping()

    # Transcribe with Whisper
    asr = load_whisper()
    transcription = transcribe_audio(asr, str(query_audio_path))
    print(f"Transcription: {transcription}")

    # Embed query audio and search
    q_emb = embed_query_audio(query_audio_path)
    q_emb = np.expand_dims(q_emb, axis=0)

    # only ask for as many neighbors as exist to avoid -1 indices
    k = min(10, max(1, int(index.ntotal)))
    scores, ids = index.search(q_emb, k)

    print(f"Top {k} matches for {query_audio_path}:")
    for rank, (idx_raw, score) in enumerate(zip(ids[0], scores[0]), 1):
        idx = int(idx_raw)  # numpy ints -> python int
        if idx < 0 or idx not in id2path:
            print(f"{rank}. <no match> | Score: {score:.4f}")
        else:
            print(f"{rank}. {id2path[idx]} | Score: {score:.4f}")

if __name__ == "__main__":
    main()
