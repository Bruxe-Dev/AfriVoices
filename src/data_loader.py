from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

try:
    from datasets import load_dataset, IterableDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `datasets` library is required. Install it with: pip install datasets"
    ) from exc

from preprocessing import clean_text, load_and_resample_audio, TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


LANGUAGE_REPO_MAP: Dict[str, Dict[str, str]] = {
    "swa": {"repo_id": "DigitalUmuganda/Afrivoice_Swahili", "config": "default", "split": "train"},
    "kik": {"repo_id": "MCAA1-MSU/anv_data_ke", "config": "kik", "split": "train"},
    "luo": {"repo_id": "MCAA1-MSU/anv_data_ke", "config": "luo", "split": "train"},
    "som": {"repo_id": "MCAA1-MSU/anv_data_ke", "config": "som", "split": "train"},
    "kln": {"repo_id": "MCAA1-MSU/anv_data_ke", "config": "kln", "split": "train"},
    "mas": {"repo_id": "MCAA1-MSU/anv_data_ke", "config": "mas", "split": "train"},
}

# Ordered by priority: the first matching, non-empty key wins.
TRANSCRIPT_KEY_CANDIDATES: List[str] = [
    "sentence",
    "transcription",
    "actualSentence",
    "transcript",
]

AUDIO_KEY_CANDIDATES: List[str] = ["audio", "speech", "wav"]


def extract_transcript(sample: Dict[str, Any]) -> str:
    row_type = sample.get("type")
    if row_type is not None:
        preferred_key = "actualSentence" if row_type in (True, "scripted") else "transcript"
        value = sample.get(preferred_key)
        if isinstance(value, str) and value.strip():
            return value

    for key in TRANSCRIPT_KEY_CANDIDATES:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def extract_audio_field(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in AUDIO_KEY_CANDIDATES:
        value = sample.get(key)
        if isinstance(value, dict) and "array" in value and "sampling_rate" in value:
            return value
    return None


def stream_language_dataset(lang_code: str) -> IterableDataset:
    if lang_code not in LANGUAGE_REPO_MAP:
        raise ValueError(
            f"Unknown language code '{lang_code}'. "
            f"Expected one of {list(LANGUAGE_REPO_MAP.keys())}."
        )

    repo_info = LANGUAGE_REPO_MAP[lang_code]
    logger.info(
        "Opening stream for lang=%s repo=%s config=%s split=%s",
        lang_code, repo_info["repo_id"], repo_info["config"], repo_info["split"],
    )
    dataset = load_dataset(
        repo_info["repo_id"],
        repo_info["config"],
        split=repo_info["split"],
        streaming=True,
    )
    return dataset


def iter_processed_samples(lang_code: str) -> Iterator[Dict[str, Any]]:
    raw_stream = stream_language_dataset(lang_code)

    for raw_sample in raw_stream:
        raw_text = extract_transcript(raw_sample)
        if not raw_text:
            logger.warning("[%s] Skipping row with no usable transcript key.", lang_code)
            continue

        audio_field = extract_audio_field(raw_sample)
        if audio_field is None:
            logger.warning("[%s] Skipping row with no usable audio field.", lang_code)
            continue

        try:
            audio_array, target_sr = load_and_resample_audio(
                audio_array=np.asarray(audio_field["array"]),
                original_sample_rate=int(audio_field["sampling_rate"]),
            )
        except ValueError as exc:
            logger.warning("[%s] Skipping row due to audio error: %s", lang_code, exc)
            continue

        yield {
            "lang_code": lang_code,
            "raw_text": raw_text,
            "clean_text": clean_text(raw_text),
            "original_sample_rate": int(audio_field["sampling_rate"]),
            "audio_array": audio_array,
            "target_sample_rate": target_sr,
        }


if __name__ == "__main__":
    SAMPLES_PER_LANGUAGE = 2

    for lang_code in LANGUAGE_REPO_MAP:
        logger.info("=" * 70)
        logger.info("LANGUAGE: %s", lang_code.upper())
        logger.info("=" * 70)

        try:
            sample_iter = iter_processed_samples(lang_code)
        except Exception as exc:  # noqa: BLE001 - top-level diagnostic guard
            logger.error("[%s] Failed to open stream: %s", lang_code, exc)
            continue

        count = 0
        for sample in sample_iter:
            count += 1
            logger.info("-" * 70)
            logger.info("Sample #%d for %s", count, lang_code)
            logger.info("  Original text : %r", sample["raw_text"])
            logger.info("  Cleaned text  : %r", sample["clean_text"])
            logger.info("  Original SR   : %d Hz", sample["original_sample_rate"])
            logger.info("  Target SR     : %d Hz", sample["target_sample_rate"])
            logger.info("  Audio shape   : %s", sample["audio_array"].shape)
            if count >= SAMPLES_PER_LANGUAGE:
                break

        if count == 0:
            logger.warning("[%s] No valid samples were found in the first batch streamed.", lang_code)