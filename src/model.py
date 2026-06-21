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
    logger.info("PHASE 2 BASELINE MODEL DRY RUN (XLS-R-300M fine-tuning setup)")
    logger.info("=" * 70)

    vocab_size, pad_token_id = load_vocab_size_from_file(VOCAB_PATH)

    logger.info("Building model from pretrained checkpoint '%s'...", PRETRAINED_CHECKPOINT)
    logger.info(
        "NOTE: you will see a warning listing 'lm_head.weight'/'lm_head.bias' as "
        "newly initialized. This is expected - the base checkpoint has no CTC "
        "head at all, so this part is created fresh, sized to our vocab_size. "
        "Everything else (the encoder) loads real pretrained weights."
    )
    model = build_pretrained_model(vocab_size=vocab_size, pad_token_id=pad_token_id)
    model.eval()  

    feature_extractor = build_feature_extractor()

    param_stats = count_parameters(model)
    logger.info("-" * 70)
    logger.info("MEASURED PARAMETER COUNT (not an estimate):")
    logger.info("  Total params      : %s", f"{param_stats['total_params']:,}")
    logger.info("  Trainable params  : %s", f"{param_stats['trainable_params']:,}  (feature encoder is frozen)")
    logger.info("  Estimated fp32 RAM: %.1f MB (parameters only, no activations/optimizer state)",
                param_stats["estimated_fp32_mb"])
    logger.info("  Constraint check  : %s",
                "PASS - under 1B hard cap" if param_stats["total_params"] < 1_000_000_000
                else "FAIL - exceeds 1B hard cap")
    logger.info(
        "  Headroom note     : this uses up most of the soft 300-500M baseline "
        "target on its own - see module docstring re: quantize/distill for "
        "the final edge deployment artifact, separate from this training-time model."
    )
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