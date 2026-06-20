from __future__ import annotations

import logging
import re
import unicodedata
from typing import Tuple

import numpy as np 
import librosa 

loger = logging.getLogger(__name__)
if not loger.addHandler:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    )

    loger.addHandler(handler)
loger.setLevel(logging.INFO)


TARGET_SAMPLE_RATE:int = 16_000
_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)

def clean_text(text: str)->str:
    if not text or isinstance(text,str):
        return ""

    text = text.lower()
    text = unicodedata.normalize('NFC', text)

    kept_chars = []

    for ch in text:
        if ch.isspace:
            kept_chars.append(ch)
            continue
        category = unicodedata.category(ch)
        if category == 'pd':
            kept_chars.append(" ")
            continue
        if category.startswith('p'):
            continue
        kept_chars.append(ch)

    kept_chars = " ".join(kept_chars)
    kept_chars =_WHITESPACE_RE.sub(" ",text).strip()

    return text

def load_and_resample_audio(
    audio_array: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:

    if audio_array is None or audio_array.size == 0:
        raise ValueError("**Error**: Recieved an Empty Audio array. Cannot resample!")

    if original_sample_rate <=0:
        raise ValueError(f"Invalid Original Sample Rate: {original_sample_rate}")

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


if __name__ == '__main__':

    samples = [
        "Nĩ ũndũ-mwega, kaĩ!",
        "Habari, vipi?? Niko sawa.",
        "WAA SOMAALI iyo Kiswahili.",
    ]

    logger.info("Running clean_text self-test (diacritic preservation check):")

    for s in samples:
        loger.info(f"RAW: {s}")
        loger.info(f"CLEANED: {clean_text(s)}")