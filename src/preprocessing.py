from __future__ import annotations

import logging
import re
import unicodedata
from typing import Tuple

import numpy as np
import librosa

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


TARGET_SAMPLE_RATE: int = 16_000  # Hz. Whisper/wav2vec2-family encoders expect this.


_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def clean_text(text: str, preserve_chars: str = "") -> str:

    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = unicodedata.normalize("NFC", text)

    kept_chars = []
    for ch in text:
        if ch.isspace():
            kept_chars.append(ch)
            continue
        if ch in preserve_chars:
            kept_chars.append(ch)
            continue
        category = unicodedata.category(ch)
        if category.startswith("P"):
            # Any punctuation category (Pc, Pd, Pe, Pf, Pi, Po, Ps) becomes
            # a space, never a deletion - see step 4 above for why.
            kept_chars.append(" ")
            continue
        kept_chars.append(ch)

    text = "".join(kept_chars)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_and_resample_audio(
    audio_array: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    if audio_array is None or audio_array.size == 0:
        raise ValueError("Received an empty audio array; cannot resample.")
    if original_sample_rate <= 0:
        raise ValueError(f"Invalid original_sample_rate: {original_sample_rate}")

    # librosa.to_mono expects shape (channels, samples); collapse to 1-D if needed.
    if audio_array.ndim > 1:
        mono_array = librosa.to_mono(audio_array.astype(np.float32))
    else:
        mono_array = audio_array.astype(np.float32)

    if original_sample_rate != target_sample_rate:
        mono_array = librosa.resample(
            mono_array,
            orig_sr=original_sample_rate,
            target_sr=target_sample_rate,
        )

    return mono_array.astype(np.float32), target_sample_rate


if __name__ == "__main__":

    samples = [
        "Nĩ ũndũ-mwega, kaĩ!",
        "Habari, vipi?? Niko sawa.",
        "WAA SOMAALI iyo Kiswahili.",
        "blight,ak eng' nematodes",  # real failing case from the Kenyan corpus
    ]
    logger.info("Running clean_text self-test (diacritic preservation check):")
    for s in samples:
        logger.info("  RAW:     %r", s)
        logger.info("  CLEANED: %r", clean_text(s))

    logger.info("-" * 70)
    logger.info("Kalenjin apostrophe decision point - default vs. preserved:")
    kln_sample = "manmegei kitit nitok ak ng'ung'unyek che ng'woneen"
    logger.info("  RAW                          : %r", kln_sample)
    logger.info("  CLEANED (apostrophe stripped): %r", clean_text(kln_sample))
    logger.info("  CLEANED (apostrophe kept)    : %r", clean_text(kln_sample, preserve_chars="'"))
    logger.info(
        "  NOTE: Teammate 1 (Linguistic Lead) should confirm whether the "
        "apostrophe in Kalenjin words like \"ng'oen\" is phonemically "
        "meaningful. If so, pass preserve_chars=\"'\" when cleaning Kalenjin "
        "text everywhere in the pipeline."
    )