import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

REPO_ID = "MCAA1-MSU/anv_data_ke"
LANG = "kik"
SPLIT = "train"
DATA_TYPE = "scripted"

fs = HfFileSystem()
parquet_path = f"datasets/{REPO_ID}/{LANG}/{SPLIT}/{DATA_TYPE}/audios/{SPLIT}_{DATA_TYPE}_000.parquet"

print("=" * 70)
print(f"Reading SCHEMA ONLY (no full download) of: {parquet_path}")
print("=" * 70)
try:
    with fs.open(parquet_path, "rb") as f:
        schema = pq.read_schema(f)
    print("Columns and types:")
    for field in schema:
        print(f"  {field.name}: {field.type}")
except Exception as exc:
    print("Could not read schema:", exc)
    print("Trying to list the audios/ folder to confirm the exact filename pattern...")
    try:
        files = fs.ls(f"datasets/{REPO_ID}/{LANG}/{SPLIT}/{DATA_TYPE}/audios", detail=False)
        for f_name in files[:5]:
            print("  ", f_name)
    except Exception as exc2:
        print("  Listing also failed:", exc2)

print()
print("=" * 70)
print("Reading first 2 ROWS ONLY of the same parquet shard (still avoids full download)")
print("=" * 70)
try:
    with fs.open(parquet_path, "rb") as f:
        pf = pq.ParquetFile(f)
        first_batch = next(pf.iter_batches(batch_size=2))
        df_preview = first_batch.to_pandas()
        # Don't print raw audio bytes if present - just show non-bytes columns.
        for col in df_preview.columns:
            sample_val = df_preview[col].iloc[0]
            if isinstance(sample_val, (bytes, bytearray)):
                print(f"  {col}: <bytes, len={len(sample_val)}>")
            else:
                print(f"  {col}: {sample_val!r}")
except Exception as exc:
    print("Could not read sample rows:", exc)

print()
print("=" * 70)
print(f"Reading full transcripts.csv for {LANG}/{SPLIT}/{DATA_TYPE} (small file)")
print("=" * 70)
try:
    csv_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"{LANG}/{SPLIT}/{DATA_TYPE}/files/transcripts.csv",
    )
    df = pd.read_csv(csv_path)
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string())
except Exception as exc:
    print("Could not download transcripts.csv:", exc)