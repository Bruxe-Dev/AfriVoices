"""
verify_swahili_shard.py
=========================
Downloads manifest_0.jsonl and audio_0.tar.xz for health_swahili_train,
then confirms:
  1. The real field names in a manifest row.
  2. Whether the "audio_filepath" value in a manifest row matches an actual
     member name inside the corresponding tar.xz archive.

This is the last verification step before writing production extraction
code, so we don't guess at the join key.
"""

import json
import tarfile
from huggingface_hub import hf_hub_download

REPO_ID = "DigitalUmuganda/Afrivoice_Swahili"
FOLDER = "health_swahili_train"

print("Downloading manifest_0.jsonl ...")
manifest_path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=f"{FOLDER}/manifest_0.jsonl",
)

rows = []
with open(manifest_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"Loaded {len(rows)} manifest rows. First row:")
print(json.dumps(rows[0], indent=2, ensure_ascii=False))

print()
print("Downloading audio_0.tar.xz (this may take a moment)...")
audio_tar_path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=f"{FOLDER}/audio/audio_0.tar.xz",
)

print("Listing first 5 members of the tar archive:")
with tarfile.open(audio_tar_path, mode="r:xz") as tf:
    members = tf.getnames()
    for m in members[:5]:
        print(" ", m)

print()
target_filename = rows[0].get("audio_filepath")
print(f"Manifest row's audio_filepath: {target_filename}")
match = [m for m in members if target_filename and target_filename in m]
print("Matching tar member(s):", match if match else "NO MATCH FOUND")