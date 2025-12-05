import pandas as pd
from pathlib import Path

# INPUT CSVs
KEYWORDS_CSV = "data/clothov2/keywords.csv"
CAPTIONS_CSV = "data/clothov2/development.csv"

# OUTPUT CSV
OUTPUT = "data/metadata.csv"

# Root path to audio files
AUDIO_ROOT = "data/clothov2/development/development/"

# Load both CSVs
df_kw = pd.read_csv(KEYWORDS_CSV, encoding="utf-8", encoding_errors="ignore")
df_cap = pd.read_csv(CAPTIONS_CSV, encoding="utf-8", encoding_errors="ignore")


# ------ CLEAN OS KEYWORDS ------
df_kw["keywords_clean"] = (
    df_kw["keywords"]
    .astype(str)
    .str.replace(";", " ", regex=False)
    .str.replace(",", " ", regex=False)
    .str.lower()
    .str.replace(" +", " ", regex=True)
    .str.strip()
)

# ------ CLEAN FS CAPTIONS ------
caption_cols = ["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]

for c in caption_cols:
    df_cap[c] = df_cap[c].astype(str).str.lower()

# Merge captions into one field
df_cap["captions_clean"] = df_cap[caption_cols].apply(lambda x: " ".join(x), axis=1)

# ------ MERGE BOTH DATASETS ------
df = df_kw.merge(df_cap, on="file_name", how="inner")

# Path column
df["path"] = AUDIO_ROOT + df["file_name"]

# ------ FINAL METADATA = OS + FS ------
df["metadata"] = df["keywords_clean"] + " " + df["captions_clean"]

# Save final metadata.csv
df[["path", "metadata"]].to_csv(OUTPUT, index=False)

print("Saved metadata.csv with OS + FS combined")
print(df[["path", "metadata"]].head())
