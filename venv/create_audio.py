import numpy as np
import soundfile as sf
from pathlib import Path

output_dir = Path("data/audio")
output_dir.mkdir(parents=True, exist_ok=True)

sr = 16000

# Create different test sounds
sounds = [
    ("beep_440hz", 440, 2.0),      # 440Hz beep for 2 seconds
    ("beep_880hz", 880, 2.5),      # 880Hz beep for 2.5 seconds  
    ("low_tone", 220, 3.0),        # 220Hz low tone for 3 seconds
    ("chirp", None, 2.0),          # Frequency sweep for 2 seconds
]

for name, freq, duration in sounds:
    t = np.linspace(0, duration, int(sr * duration))
    
    if freq is not None:
        # Simple sine wave
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
    else:
        # Frequency sweep (chirp)
        f_start, f_end = 200, 1000
        audio = 0.3 * np.sin(2 * np.pi * (f_start + (f_end - f_start) * t / duration) * t)
    
    output_path = output_dir / f"{name}.wav"
    sf.write(output_path, audio, sr)
    print(f"✓ Created {output_path}")

print("\nTest audio files created successfully!")