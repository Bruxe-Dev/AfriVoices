"""
diagnose_som_decode_failures.py
=================================
Streams the Somali scripted/train shard until it hits a few of the same
"Format not recognised" failures seen in the real run, then inspects the
RAW BYTES of those specific files to determine what they actually are:
  - Genuinely corrupted/truncated data (likely unrecoverable), or
  - A real audio format that just isn't plain WAV despite being named
    ".wav" (likely recoverable via a different decoder, e.g. PyAV).

This does NOT modify kenyan_loader.py - it's read-only diagnosis first,
fix second.
"""

import io

REPO_ID = "MCAA1-MSU/anv_data_ke"
LANG = "som"
SPLIT = "train"
DATA_TYPE = "scripted"

# Common file format "magic numbers" (first few bytes), for identification.
MAGIC_SIGNATURES = {
    b"RIFF": "WAV (RIFF container)",
    b"OggS": "Ogg (likely Opus or Vorbis audio)",
    b"\x1a\x45\xdf\xa3": "WebM/Matroska container",
    b"ID3": "MP3 (with ID3 tag)",
    b"\xff\xfb": "MP3 (no ID3 tag, raw frame sync)",
    b"\xff\xf3": "MP3 (MPEG-2)",
    b"fLaC": "FLAC",
}


def identify_format(raw_bytes: bytes) -> str:
    for magic, name in MAGIC_SIGNATURES.items():
        if raw_bytes.startswith(magic):
            return name
    return f"UNKNOWN (first 16 bytes: {raw_bytes[:16]!r})"


def run_diagnosis() -> None:
    """Run the live diagnosis against the real Somali shard. Network required."""
    from datasets import load_dataset, Audio
    from huggingface_hub import HfApi
    import soundfile as sf

    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    prefix = f"{LANG}/{SPLIT}/{DATA_TYPE}/audios/"
    shard_files = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".parquet"))
    data_files = [f"hf://datasets/{REPO_ID}/{f}" for f in shard_files]

    print(f"Streaming {len(data_files)} shard(s) for {LANG}/{SPLIT}/{DATA_TYPE}...")
    ds = load_dataset("parquet", data_files=data_files, streaming=True, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    found_failures = 0
    checked = 0
    MAX_FAILURES_TO_INSPECT = 5

    for row in ds:
        checked += 1
        audio_field = row.get("audio")
        if not audio_field or not audio_field.get("bytes"):
            continue

        raw_bytes = audio_field["bytes"]
        try:
            sf.read(io.BytesIO(raw_bytes), dtype="float32")
            continue  # Decoded fine, not one of our failures - skip.
        except Exception as exc:
            found_failures += 1
            print("=" * 70)
            print(f"FAILURE #{found_failures}: {row.get('filename')}")
            print(f"  soundfile error: {exc}")
            print(f"  Byte length     : {len(raw_bytes)}")
            print(f"  Detected format : {identify_format(raw_bytes)}")

            # Try PyAV as a fallback decoder, in case it's a different real
            # audio format that soundfile just doesn't support.
            try:
                import av
                container = av.open(io.BytesIO(raw_bytes))
                audio_stream = container.streams.audio[0]
                frame_count = sum(1 for _ in container.decode(audio_stream))
                container.close()
                print(f"  PyAV fallback   : SUCCESS - decoded {frame_count} frame(s), "
                      f"native codec = {audio_stream.codec_context.name}, "
                      f"sample_rate = {audio_stream.rate}")
            except Exception as av_exc:
                print(f"  PyAV fallback   : ALSO FAILED - {av_exc}")
                print("  --> This file may be genuinely corrupted/unrecoverable.")

            if found_failures >= MAX_FAILURES_TO_INSPECT:
                break

    print("=" * 70)
    print(f"Checked {checked} rows total, inspected {found_failures} failure(s).")


if __name__ == "__main__":
    run_diagnosis()