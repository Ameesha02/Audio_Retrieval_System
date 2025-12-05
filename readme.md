# Audio Retrieval System (Audio + Metadata Fusion)  
**Hybrid Retrieval using YAMNet, Whisper ASR, Sentence-BERT & Late Fusion**

This repository implements a hybrid Audio Retrieval System that retrieves the most relevant audio files based on an audio query, combining:

- **Audio Embeddings** (YAMNet – TF Hub)  
- **Metadata Embeddings** (Sentence-BERT)  
- **Captions + FS Keywords**  
- **Whisper ASR** for speech transcription  
- **Late Fusion Ranking**  
- **FAISS** for fast vector search  
- **Evaluation Metrics**: Recall@K, mAP@10, nDCG  

The system follows the approach from the paper:  
*"Fusing Audio and Metadata Embeddings Improves Language-Based Audio Retrieval"* – EUSIPCO 2024

---
## 🔧 Installation

```bash
git clone https://github.com/Ameesha02/Audio_Retrieval_System
cd Audio_Retrieval_System/venv

pip install -r requirements.txt
```

## 📝 Metadata Pipeline (Captions + Keywords)

This project uses two metadata sources:

Full-sentence captions (OS metadata)
From Clotho dataset: caption_1, caption_2, ..., caption_5

Closed-set keywords (FS metadata)
From keywords.csv: file_name, keywords

Merged into a single metadata field:

path,metadata
data/.../rain.wav,"A heavy storm sound... | keywords: rain storm thunder"

Generate Metadata
```bash
python prepare_metadata_os_fs.py
```

##  🔨 Building Audio Index (YAMNet)
```bash
python audio_index.py
## Creates:
artifacts/faiss_yamnet.index
```

## 🧠 Building Metadata Index (Sentence-BERT)
```bash
python build_metadata_index.py

# Creates:
artifacts/faiss_metadata.index
artifacts/id_to_path_meta.txt
artifacts/id_to_path.txt
```

## 🚀 Running the Streamlit App
Launch the retrieval system:
```bash
streamlit run app_fusion.py
```
### Features

- Upload audio query

- Whisper transcribes speech

- YAMNet extracts audio embedding

- SBERT embeds text

- Independent audio + metadata search

- Weighted late fusion

- Interactive audio playback

- Top-K ranked results
 ## Evaluation
Run the evaluation with:
```bash
python fusion_eval.py
```

This will compute and print:

- Recall@5, Recall@10  
- mAP@10  
- nDCG@10  

For:

- Audio retrieval  
- Metadata retrieval  
- Fusion retrieval  

Results are saved to `fusion_evaluation_results.csv`.

## Requirements

- Python >= 3.10  
- tensorflow  
- tensorflow-hub  
- sentence-transformers  
- whisper  
- faiss-cpu  
- numpy  
- pandas  
- soundfile  
- resampy  
- streamlit  
- tqdm  

## 📚 Reference

Primus, P., & Widmer, G. (2024).
Fusing Audio and Metadata Embeddings Improves Language-Based Audio Retrieval.
EUSIPCO 2024.

## Usage

1. Prepare your dataset and queries.  
2. Run the retrieval pipeline scripts as needed.  
3. Run `python fusion_eval.py` to evaluate retrieval performance.  
4. Check the `fusion_evaluation_results.csv` for metric results.

## Team 
- Ameesha Patel(252IT003)
- Dolly Chauhan(252IT007)
  

