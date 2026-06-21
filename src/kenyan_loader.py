from __future__ import annotations

import io
import logging
from typing import Any, Dict, Iterator, List

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from huggingface_hub import HfApi

from preprocessing import clean_text, load_and_resample_audio, TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

REPO_ID = "MCAA1-MSU/anv_data_ke"
VALID_LANGS = ("kik", "kln", "luo", "mas", "som")
VALID_SPLITS = ("train", "dev", "dev_test")
VALID_TYPES = ("scripted", "unscripted")

LANGUAGE_PRESERVE_CHARS: Dict[str, str] = {
    "kln": "'",
    "luo": "'",
}


def list_shard_files(lang: str, split: str, data_type: str) -> List[str]:

    if lang not in VALID_LANGS:
        raise ValueError(f"Unknown lang '{lang}'. Expected one of {VALID_LANGS}.")
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {VALID_SPLITS}.")
    if data_type not in VALID_TYPES:
        raise ValueError(f"Unknown data_type '{data_type}'. Expected one of {VALID_TYPES}.")

    prefix = f"{lang}/{split}/{data_type}/audios/"
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    matching = [f for f in all_files if f.startswith(prefix) and f.endswith(".parquet")]

    if not matching:
        raise ValueError(
            f"No parquet shards found under '{prefix}'. "
            f"This combination may genuinely have no data."
        )

    return [f"hf://datasets/{REPO_ID}/{f}" for f in sorted(matching)]


def decode_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:

    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # Collapse multi-channel to mono.
    return data.astype(np.float32), sample_rate


def iter_language_samples(
    lang: str,
    split: str = "train",
    data_types: tuple = VALID_TYPES,
) -> Iterator[Dict[str, Any]]:

    data_files: List[str] = []
    for data_type in data_types:
        try:
            data_files.extend(list_shard_files(lang, split, data_type))
        except ValueError as exc:
            logger.warning("[%s/%s/%s] %s", lang, split, data_type, exc)

    if not data_files:
        logger.error("[%s/%s] No shards found for any requested data_type.", lang, split)
        return

    logger.info("[%s/%s] Streaming %d shard file(s)...", lang, split, len(data_files))
    ds = load_dataset("parquet", data_files=data_files, streaming=True, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    yielded_count = 0
    for row in ds:
        raw_text = row.get("transcription", "") or ""
        if not raw_text:
            logger.warning("[%s] Skipping row with empty transcription (filename=%s)", lang, row.get("filename"))
            continue

        audio_field = row.get("audio")
        if not audio_field or not audio_field.get("bytes"):
            logger.warning("[%s] Skipping row with no audio bytes (filename=%s)", lang, row.get("filename"))
            continue

        try:
            samples, native_sr = decode_wav_bytes(audio_field["bytes"])
            audio_array, target_sr = load_and_resample_audio(samples, native_sr, TARGET_SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the stream
            logger.warning("[%s] Failed to decode audio for %s: %s", lang, row.get("filename"), exc)
            continue

        yielded_count += 1
        yield {
            "lang_code": lang,
            "raw_text": raw_text,
            "clean_text": clean_text(raw_text, preserve_chars=LANGUAGE_PRESERVE_CHARS.get(lang, "")),
            "audio_array": audio_array,
            "target_sample_rate": target_sr,
            "data_type": row.get("type"),
            "domain": row.get("domain"),
            "filename": row.get("filename"),
        }

    logger.info("[%s/%s] Yielded %d processed samples total.", lang, split, yielded_count)


if __name__ == "__main__":
    SAMPLES_PER_LANGUAGE = 3

    for lang in VALID_LANGS:
        logger.info("=" * 70)
        logger.info("LANGUAGE: %s", lang.upper())
        logger.info("=" * 70)

        count = 0
        try:
            for sample in iter_language_samples(lang, split="train", data_types=("scripted",)):
                count += 1
                logger.info("-" * 70)
                logger.info("Sample #%d for %s", count, lang)
                logger.info("  Raw text   : %r", sample["raw_text"][:100])
                logger.info("  Clean text : %r", sample["clean_text"][:100])
                logger.info("  Audio shape: %s", sample["audio_array"].shape)
                logger.info("  Target SR  : %d", sample["target_sample_rate"])
                logger.info("  Domain     : %s", sample["domain"])
                if count >= SAMPLES_PER_LANGUAGE:
                    break
        except Exception as exc:  # noqa: BLE001 - diagnostic guard
            logger.error("[%s] Failed: %s", lang, exc)

        if count == 0:
            logger.warning("[%s] No valid samples were yielded.", lang)