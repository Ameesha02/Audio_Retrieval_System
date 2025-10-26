# utils_paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Audio index artifacts
AUDIO_INDEX_PATH = ARTIFACTS_DIR / "faiss_yamnet.index"
AUDIO_MAPPING_PATH = ARTIFACTS_DIR / "id_to_path.txt"

# Metadata index artifacts
META_INDEX_PATH = ARTIFACTS_DIR / "faiss_metadata.index"
META_MAPPING_PATH = ARTIFACTS_DIR / "id_to_path_meta.txt"

# Metadata CSV
META_CSV_PATH = DATA_DIR / "metadata.csv"
print(f"Project root directory: {PROJECT_ROOT}")