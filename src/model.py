from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

TARGET_SAMPLE_RATE = 16_000


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

    logger.info("=" * 70)
    logger.info("PHASE 2 BASELINE MODEL DRY RUN")
    logger.info("=" * 70)

    vocab_size, pad_token_id = load_vocab_size_from_file(VOCAB_PATH)

    logger.info("Building config (vocab_size=%d, pad_token_id=%d)...", vocab_size, pad_token_id)
    config = build_baseline_config(vocab_size=vocab_size, pad_token_id=pad_token_id)

    logger.info("Instantiating Wav2Vec2ForCTC model (random init, no pretrained weights)...")
    model = Wav2Vec2ForCTC(config)
    model.eval()  # Dry run only - no training happening here.

    feature_extractor = build_feature_extractor()

    param_stats = count_parameters(model)
    logger.info("-" * 70)
    logger.info("MEASURED PARAMETER COUNT (not an estimate):")
    logger.info("  Total params      : %s", f"{param_stats['total_params']:,}")
    logger.info("  Trainable params  : %s", f"{param_stats['trainable_params']:,}")
    logger.info("  Estimated fp32 RAM: %.1f MB (parameters only, no activations/optimizer state)",
                param_stats["estimated_fp32_mb"])
    logger.info("  Constraint check  : %s",
                "PASS - well under 500M params" if param_stats["total_params"] < 500_000_000
                else "FAIL - exceeds 500M params, reduce hidden_size/num_hidden_layers")
    logger.info("-" * 70)

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

    logger.info("Running model forward pass...")
    with torch.no_grad():
        outputs = model(
            input_values=inputs["input_values"],
            attention_mask=inputs["attention_mask"],
        )
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
    logger.info("=" * 70)
    logger.info("Dry run complete.")