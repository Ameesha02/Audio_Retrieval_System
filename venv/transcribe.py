import whisper

def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

def transcribe_audio(asr_model, audio_path):
    result = asr_model.transcribe(audio_path)
    return result.get("text", "").strip()