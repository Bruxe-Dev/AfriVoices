from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import psutil
import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

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

TARGET_SAMPLE_RATE = 16_000


PRETRAINED_CHECKPOINT = "facebook/wav2vec2-xls-r-300m"


def build_pretrained_model(vocab_size: int, pad_token_id: int = 0) -> Wav2Vec2ForCTC:
    logger.info("Loading pretrained checkpoint '%s' (this downloads ~1.2GB on first run)...", PRETRAINED_CHECKPOINT)
    model = Wav2Vec2ForCTC.from_pretrained(
        PRETRAINED_CHECKPOINT,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        ignore_mismatched_sizes=True,  # The checkpoint has no lm_head at all; this allows a fresh one to be created.
        use_safetensors=True,  # Avoid downloading BOTH pytorch_model.bin and model.safetensors on first run.
    )
    model.freeze_feature_encoder()
    logger.info("Feature encoder frozen (standard practice for low-resource fine-tuning).")
    return model


def build_baseline_config(vocab_size: int, pad_token_id: int = 0) -> Wav2Vec2Config:

    config = Wav2Vec2Config(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        pad_token_id=pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
    )
    return config


def build_feature_extractor() -> Wav2Vec2FeatureExtractor:

    return Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=TARGET_SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )


def load_vocab_size_from_file(vocab_path: str, fallback: int = 128) -> tuple[int, int]:

    path = Path(vocab_path)
    if not path.exists():
        logger.warning(
            "vocab.json not found at '%s'. Using fallback vocab_size=%d. "
            "Run tokenizer.py first to build the real vocabulary, then "
            "re-run this script for an accurate model size.",
            vocab_path, fallback,
        )
        return fallback, 0

    payload = json.loads(path.read_text(encoding="utf-8"))
    vocab_size = payload["vocab_size"]
    pad_token_id = payload["token_to_id"]["<pad>"]
    logger.info("Loaded real vocab_size=%d, pad_token_id=%d from %s", vocab_size, pad_token_id, vocab_path)
    return vocab_size, pad_token_id


def get_process_rss_mb() -> float:

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    estimated_mb = total * 4 / (1024 ** 2)  # 4 bytes per fp32 parameter
    return {
        "total_params": total,
        "trainable_params": trainable,
        "estimated_fp32_mb": estimated_mb,
    }


if __name__ == "__main__":
    VOCAB_PATH = "vocab.json"
    REAL_TIME_FACTOR_LIMIT = 2.0  # Hard constraint: must process audio in <= 2x its duration.
    NUM_TIMING_RUNS = 5  # Average over several runs - a single forward pass can be noisy.

    logger.info("=" * 70)
    logger.info("PHASE 2 BASELINE MODEL DRY RUN (XLS-R-300M fine-tuning setup)")
    logger.info("=" * 70)

    rss_before_load = get_process_rss_mb()
    logger.info("Process RSS before model load: %.1f MB", rss_before_load)

    vocab_size, pad_token_id = load_vocab_size_from_file(VOCAB_PATH)

    logger.info("Building model from pretrained checkpoint '%s'...", PRETRAINED_CHECKPOINT)
    logger.info(
        "NOTE: you will see a warning listing 'lm_head.weight'/'lm_head.bias' as "
        "newly initialized. This is expected - the base checkpoint has no CTC "
        "head at all, so this part is created fresh, sized to our vocab_size. "
        "Everything else (the encoder) loads real pretrained weights."
    )
    model = build_pretrained_model(vocab_size=vocab_size, pad_token_id=pad_token_id)
    model.eval()  # Dry run only - no training happening here.

    feature_extractor = build_feature_extractor()

    rss_after_load = get_process_rss_mb()

    param_stats = count_parameters(model)
    logger.info("-" * 70)
    logger.info("MEASURED PARAMETER COUNT (not an estimate):")
    logger.info("  Total params      : %s", f"{param_stats['total_params']:,}")
    logger.info("  Trainable params  : %s", f"{param_stats['trainable_params']:,}  (feature encoder is frozen)")
    logger.info("  Constraint check  : %s",
                "PASS - under 1B hard cap" if param_stats["total_params"] < 1_000_000_000
                else "FAIL - exceeds 1B hard cap")
    logger.info("-" * 70)
    logger.info("MEASURED PROCESS RAM (real RSS, not a parameter-count estimate):")
    logger.info("  RSS before model load : %.1f MB", rss_before_load)
    logger.info("  RSS after model load   : %.1f MB", rss_after_load)
    logger.info("  Delta from loading model: %.1f MB", rss_after_load - rss_before_load)
    logger.info("-" * 70)

    AUDIO_DURATION_SECONDS = 1.0
    logger.info("Building dummy audio tensor: shape [1, 16000] (1 second @ 16kHz mono)...")
    dummy_audio = torch.randn(1, TARGET_SAMPLE_RATE)  # NOTE: random noise, shapes only - not meaningful audio.
    logger.info("  dummy_audio.shape = %s", tuple(dummy_audio.shape))

    logger.info("Running feature_extractor...")
    inputs = feature_extractor(
        dummy_audio.numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    logger.info("  inputs['input_values'].shape    = %s", tuple(inputs["input_values"].shape))
    logger.info("  inputs['attention_mask'].shape  = %s", tuple(inputs["attention_mask"].shape))

    logger.info("Running 1 warm-up forward pass (excluded from timing - first call often has extra overhead)...")
    with torch.no_grad():
        outputs = model(input_values=inputs["input_values"], attention_mask=inputs["attention_mask"])

    logger.info("Timing %d forward passes on CPU...", NUM_TIMING_RUNS)
    run_times = []
    for i in range(NUM_TIMING_RUNS):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_values=inputs["input_values"], attention_mask=inputs["attention_mask"])
        elapsed = time.perf_counter() - start
        run_times.append(elapsed)
        logger.info("  Run %d: %.3f sec", i + 1, elapsed)

    avg_time = sum(run_times) / len(run_times)
    real_time_factor = avg_time / AUDIO_DURATION_SECONDS

    rss_after_inference = get_process_rss_mb()

    logger.info("  outputs.logits.shape = %s", tuple(outputs.logits.shape))
    logger.info(
        "  Interpretation: [batch_size=%d, num_output_timesteps=%d, vocab_size=%d]",
        outputs.logits.shape[0], outputs.logits.shape[1], outputs.logits.shape[2],
    )
    logger.info(
        "  Final dimension (%d) matches tokenizer vocab_size (%d): %s",
        outputs.logits.shape[2], vocab_size,
        "MATCH" if outputs.logits.shape[2] == vocab_size else "MISMATCH - check config",
    )
    logger.info("-" * 70)
    logger.info("MEASURED CPU LATENCY (real wall-clock time, not an estimate):")
    logger.info("  Audio duration       : %.1f sec", AUDIO_DURATION_SECONDS)
    logger.info("  Avg forward pass time: %.3f sec (over %d runs)", avg_time, NUM_TIMING_RUNS)
    logger.info("  Real-time factor     : %.2fx", real_time_factor)
    logger.info(
        "  Constraint check     : %s (limit: <= %.1fx)",
        f"PASS - {real_time_factor:.2f}x" if real_time_factor <= REAL_TIME_FACTOR_LIMIT
        else f"FAIL - {real_time_factor:.2f}x exceeds limit",
        REAL_TIME_FACTOR_LIMIT,
    )
    logger.info("-" * 70)
    logger.info("MEASURED PEAK RAM (real RSS during this dry run, single 1-second clip, batch_size=1):")
    logger.info("  RSS after inference: %.1f MB", rss_after_inference)
    logger.info(
        "  Constraint check   : %s (limit: 8192 MB / 8GB)",
        "PASS" if rss_after_inference < 8192 else "FAIL - exceeds 8GB",
    )
    logger.info(
        "  CAVEAT: this measures ONE short clip in this Python process. Real "
        "deployment RAM also depends on batch size, audio length, and the "
        "specific runtime (e.g. ONNX/quantized) actually used on the Pi 4 - "
        "this number is a useful first signal, not the final edge benchmark "
        "Teammate 3 will eventually need to run on real target hardware."
    )
    logger.info("=" * 70)
    logger.info("Dry run complete.")