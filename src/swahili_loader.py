from __future__ import annotations

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import av
from huggingface_hub import HfApi, hf_hub_download

from preprocessing import clean_text, load_and_resample_audio, TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

REPO_ID = "DigitalUmuganda/Afrivoice_Swahili"
VALID_DOMAINS = ("agriculture", "education", "financial", "government", "health")
VALID_SPLITS = ("train", "dev", "test")


def list_shard_indices(domain: str, split: str) -> List[int]:
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Expected one of {VALID_DOMAINS}.")
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {VALID_SPLITS}.")

    folder = f"{domain}_swahili_{split}"
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")

    indices = []
    prefix = f"{folder}/manifest_"
    for f in all_files:
        if f.startswith(prefix) and f.endswith(".jsonl"):
            idx_str = f[len(prefix):-len(".jsonl")]
            try:
                indices.append(int(idx_str))
            except ValueError:
                logger.warning("Could not parse shard index from filename: %s", f)

    if not indices:
        raise ValueError(f"No manifest shards found under '{folder}/'. Check domain/split spelling.")

    return sorted(indices)


def decode_webm_bytes(webm_bytes: bytes) -> tuple[np.ndarray, int]:
    container = av.open(io.BytesIO(webm_bytes))
    audio_stream = container.streams.audio[0]
    sample_rate = audio_stream.rate

    frames = [frame.to_ndarray() for frame in container.decode(audio_stream)]
    container.close()

    if not frames:
        raise ValueError("No audio frames decoded from webm bytes.")

    full = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
    mono = full.mean(axis=0) if full.shape[0] > 1 else full[0]

    if mono.dtype.kind in ("i", "u"):
        max_val = np.iinfo(mono.dtype).max
        mono = mono.astype(np.float32) / max_val
    else:
        mono = mono.astype(np.float32)

    return mono, sample_rate


def iter_shard_samples(domain: str, split: str, shard_idx: int) -> Iterator[Dict[str, Any]]:
    folder = f"{domain}_swahili_{split}"
    manifest_filename = f"{folder}/manifest_{shard_idx}.jsonl"
    audio_tar_filename = f"{folder}/audio/audio_{shard_idx}.tar.xz"
    shard_basename = f"audio_{shard_idx}"

    logger.info("[%s/%s shard %d] Downloading manifest...", domain, split, shard_idx)
    manifest_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=manifest_filename)

    manifest_rows: Dict[str, Dict[str, Any]] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_filepath = row.get("audio_filepath")
            if audio_filepath:
                manifest_rows[audio_filepath] = row

    logger.info(
        "[%s/%s shard %d] Loaded %d manifest rows. Downloading audio tar...",
        domain, split, shard_idx, len(manifest_rows),
    )
    audio_tar_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=audio_tar_filename)

    yielded_count = 0
    with tarfile.open(audio_tar_path, mode="r:xz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            # Members are named "{shard_basename}/{audio_filepath}";
            # strip the shard_basename prefix to get the join key.
            member_basename = Path(member.name).name
            row = manifest_rows.get(member_basename)
            if row is None:
                continue  # Audio file with no matching manifest row; skip.

            raw_text = row.get("transcription", "") or ""
            if not raw_text:
                logger.warning("[%s] No transcription for %s; skipping.", shard_basename, member_basename)
                continue

            try:
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                webm_bytes = extracted.read()
                samples, native_sr = decode_webm_bytes(webm_bytes)
                audio_array, target_sr = load_and_resample_audio(samples, native_sr, TARGET_SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the shard
                logger.warning("[%s] Failed to decode %s: %s", shard_basename, member_basename, exc)
                continue

            yielded_count += 1
            yield {
                "raw_text": raw_text,
                "clean_text": clean_text(raw_text),
                "audio_array": audio_array,
                "target_sample_rate": target_sr,
                "duration_seconds": row.get("duration"),
                "category": row.get("category"),
                "locale": row.get("locale"),
            }

    logger.info("[%s/%s shard %d] Yielded %d processed samples.", domain, split, shard_idx, yielded_count)


def iter_domain_samples(domain: str, split: str) -> Iterator[Dict[str, Any]]:
    shard_indices = list_shard_indices(domain, split)
    logger.info("Found %d shards for %s/%s: %s", len(shard_indices), domain, split, shard_indices)
    for shard_idx in shard_indices:
        yield from iter_shard_samples(domain, split, shard_idx)


if __name__ == "__main__":

    logger.info("Running diagnostic on health_swahili_train, shard 0 only...")
    count = 0
    for sample in iter_shard_samples("health", "train", shard_idx=0):
        count += 1
        logger.info("-" * 70)
        logger.info("Sample #%d", count)
        logger.info("  Raw text   : %r", sample["raw_text"][:100])
        logger.info("  Clean text : %r", sample["clean_text"][:100])
        logger.info("  Audio shape: %s", sample["audio_array"].shape)
        logger.info("  Target SR  : %d", sample["target_sample_rate"])
        logger.info("  Duration   : %s sec", sample["duration_seconds"])
        if count >= 3:
            break