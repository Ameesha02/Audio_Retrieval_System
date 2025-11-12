import os
import sys
import numpy as np
import faiss
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import resampy
from tqdm import tqdm
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_ROOT = Path(__file__).parent.resolve()
AUDIO_DIR = PROJECT_ROOT / "data" / "clothov2" / "development" / "development"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "faiss_yamnet.index"
MAPPING_PATH = ARTIFACTS_DIR / "id_to_path.txt"
EMBEDDING_SIZE = 1024



print("="*60)
print("Starting audio indexing with YAMNet")
print("="*60)

# Load YAMNet model
print("\n1. Loading YAMNet model from TensorFlow Hub...")
try:
    yamnet_model = hub.load("/home/student/MIR_project/venv/models")
    print("   ✓ YAMNet model loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load YAMNet: {e}")
    sys.exit(1)

def load_audio_resample(file_path, target_sr=16000):
    """Load and resample audio."""
    try:
        audio, sr = sf.read(str(file_path), dtype='float32')
        print(f"   Loaded: {file_path.name} | Shape: {audio.shape} | SR: {sr}Hz")
        
        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            print(f"   Converted to mono: {audio.shape}")
        
        # Check minimum length
        duration = len(audio) / sr
        if duration < 0.1:
            print(f"   ✗ Too short ({duration:.2f}s), skipping")
            return None
        
        # Resample if needed
        if sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
            print(f"   Resampled to {target_sr}Hz")
        
        return audio
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None

def embed_audio(file_path):
    """Generate YAMNet embedding."""
    audio = load_audio_resample(file_path)
    if audio is None:
        return None
    
    try:
        scores, embeddings, spectrogram = yamnet_model(audio)
        emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
        print(f"   ✓ Generated embedding: {emb_mean.shape}")
        return emb_mean
    except Exception as e:
        print(f"   ✗ Embedding failed: {e}")
        return None

# ...existing code...
def list_audio_files(root: Path):
    """List all audio files (recursively)."""
    if not root.exists():
        return []
    files = []
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
            files.append(f)
    return sorted(files)
# ...existing code...

def main():
    print(f"Script location: {PROJECT_ROOT}")
    print(f"Looking for audio in: {AUDIO_DIR}")
    print(f"AUDIO_DIR exists? {AUDIO_DIR.exists()}")
    print(f"Contents of PROJECT_ROOT: {list(PROJECT_ROOT.iterdir())}")
    print("\n2. Creating artifacts directory...")
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    print(f"   ✓ Directory: {ARTIFACTS_DIR.absolute()}")
    
    print(f"\n3. Searching for audio files in {AUDIO_DIR.absolute()}...")
    audio_files = list_audio_files(AUDIO_DIR)
    print(f"   Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("   ✗ No audio files found!")
        print(f"   Please add .wav/.mp3/.ogg files to {AUDIO_DIR.absolute()}")
        sys.exit(1)
    
    for f in audio_files:
        print(f"   - {f.name}")
    
    print("\n4. Generating embeddings...")
    embeddings = []
    valid_files = []
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n   Processing [{i}/{len(audio_files)}]: {audio_path.name}")
        emb = embed_audio(audio_path)
        if emb is not None:
            embeddings.append(emb)
            valid_files.append(audio_path)
    
    if len(embeddings) == 0:
        print("\n✗ No valid embeddings generated!")
        print("   Check your audio files and try again.")
        sys.exit(1)
    
    embeddings = np.stack(embeddings).astype("float32")
    print(f"\n5. Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    print("\n6. Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    print("   ✓ Normalization complete")
    
    print("\n7. Building FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    index.add(embeddings)
    print(f"   ✓ Index built with {index.ntotal} vectors")
    
    print("\n8. Saving index and mappings...")
    faiss.write_index(index, str(INDEX_PATH))
    print(f"   ✓ Saved index: {INDEX_PATH}")
    
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        for i, p in enumerate(valid_files):
            f.write(f"{i}\t{p}\n")
    print(f"   ✓ Saved mapping: {MAPPING_PATH}")
    
    print("\n" + "="*60)
    print(f"SUCCESS! Indexed {len(valid_files)} audio files")
    print("="*60)
    print(f"\nYou can now run:")
    print(f"  python search_speech.py {valid_files[0]}")

if __name__ == "__main__":
    main()