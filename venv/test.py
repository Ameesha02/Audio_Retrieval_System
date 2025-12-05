# check_id_alignment.py
import pathlib

audio_map = pathlib.Path("artifacts/id_to_path.txt")
meta_map = pathlib.Path("artifacts/id_to_path_meta.txt")

print("Checking path consistency...\n")

with audio_map.open() as f1, meta_map.open() as f2:
    audio_lines = f1.readlines()
    meta_lines = f2.readlines()

print(f"Audio index entries: {len(audio_lines)}")
print(f"Metadata index entries: {len(meta_lines)}")

# Check same count
if len(audio_lines) != len(meta_lines):
    print("❌ COUNT MISMATCH → accuracy will be wrong!")
else:
    print("✓ Same number of entries")

print("\nComparing first 20 IDs...")
for i in range(min(20, len(audio_lines))):
    a_id, a_path = audio_lines[i].strip().split("\t")
    m_id, m_path = meta_lines[i].strip().split("\t")

    print(f"ID {i}:")
    print(f"  AUDIO → {a_path}")
    print(f"  META  → {m_path}")

    if a_path != m_path:
        print("  ❌ MISMATCH HERE")
        break
    else:
        print("  ✓ Match")

print("\nDONE.")
