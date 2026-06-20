"""
verify_kenyan_audio_decode.py
================================
Confirms whether load_dataset("parquet", ...) auto-decodes the "audio"
struct column into a ready {"array": ..., "sampling_rate": ...} dict, or
whether it hands back raw {"bytes": ..., "path": ...} that we'd need to
decode ourselves via soundfile.

Also confirms whether "transcription" is already present and populated
per-row, meaning the parquet shard is self-contained (no CSV join needed).
"""

from datasets import load_dataset, Audio

PARQUET_GLOB = "hf://datasets/MCAA1-MSU/anv_data_ke/kik/train/scripted/audios/train_scripted_000.parquet"

print("Opening streaming connection to one parquet shard...")
ds = load_dataset("parquet", data_files=PARQUET_GLOB, streaming=True, split="train")

# Disable automatic decoding of the audio column. This avoids requiring the
# heavy torchcodec dependency - we'll decode the raw WAV bytes ourselves
# with the much lighter `soundfile` library instead.
ds = ds.cast_column("audio", Audio(decode=False))

print("Pulling first row...")
row = next(iter(ds))

print("=" * 70)
print("Row keys:", list(row.keys()))
print("=" * 70)

audio_val = row["audio"]
print("Type of row['audio']:", type(audio_val))
if isinstance(audio_val, dict):
    print("Keys inside audio dict:", list(audio_val.keys()))
    if "bytes" in audio_val and audio_val["bytes"]:
        import soundfile as sf
        import io
        import numpy as np

        data, samplerate = sf.read(io.BytesIO(audio_val["bytes"]), dtype="float32")
        print("MANUALLY DECODED via soundfile: shape =", data.shape, " samplerate =", samplerate)
        print("path field:", audio_val.get("path"))

print()
print("transcription field:", repr(row.get("transcription"))[:200])
print("filename field:", row.get("filename"))
print("type field:", row.get("type"))
print("domain field:", row.get("domain"))