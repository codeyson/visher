# chunk_audio.py  — run from project root
import soundfile as sf
import numpy as np
from pathlib import Path

SRC_DIR    = Path("LibriSeVoc/gt")
CHUNK_SAMP = 64600
TARGET_SR  = 24000

for wav_path in SRC_DIR.glob("youtube_*.wav"):
    audio, sr = sf.read(wav_path, dtype="float32")
    
    # mix to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    # resample if needed
    if sr != TARGET_SR:
        import torchaudio.functional as TAF
        import torch
        t = torch.from_numpy(audio).unsqueeze(0)
        t = TAF.resample(t, sr, TARGET_SR)
        audio = t.squeeze(0).numpy()
    
    # slice into chunks
    n_chunks = len(audio) // CHUNK_SAMP
    for i in range(n_chunks):
        chunk = audio[i*CHUNK_SAMP : (i+1)*CHUNK_SAMP]
        # skip silent chunks
        if np.sqrt(np.mean(chunk**2)) < 0.01:
            continue
        out_path = SRC_DIR / f"{wav_path.stem}_chunk{i:04d}.wav"
        sf.write(out_path, chunk, TARGET_SR)
    
    print(f"✅ {wav_path.name} → {n_chunks} chunks")

print("Done.")