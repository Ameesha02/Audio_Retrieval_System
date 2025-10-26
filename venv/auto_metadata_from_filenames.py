# auto_metadata_from_filenames.py
import re
import pandas as pd
from pathlib import Path
from utils_paths import DATA_DIR, AUDIO_DIR, META_CSV_PATH

VALID_EXT = {".wav", ".mp3", ".ogg", ".flac"}

def humanize(name: str) -> str:
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    metas = []
    for p in AUDIO_DIR.glob("**/*"):
        if p.suffix.lower() in VALID_EXT:
            paths.append(str(p))
            base = p.stem
            desc = humanize(base)
            metas.append(desc if desc else "audio clip")
    df = pd.DataFrame({"path": paths, "metadata": metas})
    META_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(META_CSV_PATH, index=False)
    print(f"Wrote {len(df)} rows to {META_CSV_PATH}")

if __name__ == "__main__":
    main()