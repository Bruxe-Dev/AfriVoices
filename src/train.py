from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from torch.utils.data import Dataset
from transformers import Trainer, Wav2Vec2ForCTC

try:
    from . import kenyan_loader, swahili_loader
    from .collector import DataCollatorCTCWithPadding
    from .metrics import compute_metrics_fn, compute_unweighted_multilingual_wer
    from .model import build_baseline_config, build_feature_extractor, build_pretrained_model
    from .tokenizer import CharacterTokenizer
    from .train_config import build_training_arguments
    from .utils import (
        DEFAULT_MAX_CLIP_SECONDS,
        MemoryMonitorCallback,
        get_logger,
        set_global_seed,
    )
except ImportError:  # pragma: no cover - allows running the script directly from src/
    import kenyan_loader
    import swahili_loader
    from collector import DataCollatorCTCWithPadding
    from metrics import compute_metrics_fn, compute_unweighted_multilingual_wer
    from model import build_baseline_config, build_feature_extractor, build_pretrained_model
    from tokenizer import CharacterTokenizer
    from train_config import build_training_arguments
    from utils import (
        DEFAULT_MAX_CLIP_SECONDS,
        MemoryMonitorCallback,
        get_logger,
        set_global_seed,
    )

logger = get_logger(__name__)

VOCAB_PATH_DEFAULT = "vocab.json"
SWAHILI_DOMAINS = swahili_loader.VALID_DOMAINS
KENYAN_LANGS = kenyan_loader.VALID_LANGS
ALL_LANGS: tuple = ("swa",) + KENYAN_LANGS


def _swahili_iterator(split: str) -> Iterator[Dict[str, Any]]:
    """swahili_loader has no concept of 'lang_code' (it only ever serves one language)
    and is split by domain rather than by language, so this chains all five domains
    and stamps lang_code='swa' onto every row to match the schema kenyan_loader uses."""
    for domain in SWAHILI_DOMAINS:
        try:
            for sample in swahili_loader.iter_domain_samples(domain, split):
                sample["lang_code"] = "swa"
                yield sample
        except Exception as exc:  # noqa: BLE001 - one missing domain/split shouldn't kill the run
            logger.warning("Swahili domain '%s'/%s unavailable: %s", domain, split, exc)


def _get_language_iterator(lang: str, split: str) -> Iterator[Dict[str, Any]]:
    if lang == "swa":
        return _swahili_iterator(split)
    return kenyan_loader.iter_language_samples(lang, split=split)


def interleave_with_caps(
    languages: List[str],
    split: str,
    max_per_lang: Optional[int],
    max_clip_seconds: float,
) -> Iterator[Dict[str, Any]]:
    iterators = {lang: _get_language_iterator(lang, split) for lang in languages}
    counts = {lang: 0 for lang in languages}
    dropped_for_length = {lang: 0 for lang in languages}
    active = set(languages)

    while active:
        for lang in list(active):
            if max_per_lang is not None and counts[lang] >= max_per_lang:
                active.discard(lang)
                continue
            try:
                sample = next(iterators[lang])
            except StopIteration:
                active.discard(lang)
                continue

            duration = sample["audio_array"].shape[-1] / sample["target_sample_rate"]
            if duration <= 0 or duration > max_clip_seconds:
                dropped_for_length[lang] += 1
                continue

            counts[lang] += 1
            yield sample

    logger.info("[%s] Collected per language: %s", split, counts)
    if any(dropped_for_length.values()):
        logger.info(
            "[%s] Dropped for exceeding %.1fs safety cap: %s",
            split, max_clip_seconds, dropped_for_length,
        )


def collect_samples(
    languages: List[str],
    split: str,
    max_per_lang: Optional[int],
    max_clip_seconds: float,
) -> List[Dict[str, Any]]:
    samples = list(interleave_with_caps(languages, split, max_per_lang, max_clip_seconds))
    if not samples:
        raise RuntimeError(
            f"Collected zero samples for split='{split}' across languages={languages}. "
            f"Check network access to the HF Hub repos and the language/domain spelling."
        )
    return samples


# --------------------------------------------------------------------------- #
# torch Dataset wrapper
# --------------------------------------------------------------------------- #

class MultilingualASRDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: CharacterTokenizer,
        feature_extractor: Any,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        labels = self.tokenizer.encode(
            sample["clean_text"],
            lang_code=sample.get("lang_code"),
            add_bos_eos=True,
        )
        input_values = self.feature_extractor(
            sample["audio_array"],
            sampling_rate=sample.get("target_sample_rate", 16000),
            return_tensors="np",
        )["input_values"][0]
        return {
            "input_values": input_values,
            "labels": labels,
        }


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #

def build_model(model_type: str, vocab_size: int, pad_token_id: int) -> Wav2Vec2ForCTC:
    if model_type == "pretrained":
        return build_pretrained_model(vocab_size=vocab_size, pad_token_id=pad_token_id)
    config = build_baseline_config(vocab_size=vocab_size, pad_token_id=pad_token_id)
    return Wav2Vec2ForCTC(config)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AfriVoices East Africa ASR model.")
    parser.add_argument(
        "--languages", nargs="+", default=list(ALL_LANGS), choices=list(ALL_LANGS),
        help="Which of the 6 target languages to train on. Default: all six.",
    )
    parser.add_argument("--model_type", choices=["baseline", "pretrained"], default="baseline")
    parser.add_argument("--vocab_path", default=VOCAB_PATH_DEFAULT)
    parser.add_argument("--output_dir", default="./afrivoices_checkpoints")
    parser.add_argument(
        "--max_train_samples_per_lang", type=int, default=2000,
        help="Cap per language for the train split, applied during round-robin collection.",
    )
    parser.add_argument("--max_eval_samples_per_lang", type=int, default=200)
    parser.add_argument(
        "--max_clip_seconds", type=float, default=DEFAULT_MAX_CLIP_SECONDS,
        help="Safety cap: drop any single clip longer than this, to protect the 8GB RAM budget.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--ram_limit_mb", type=float, default=8192.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    logger.info("=" * 70)
    logger.info("PHASE 3: TRAINING  |  languages=%s  |  model_type=%s", args.languages, args.model_type)
    logger.info("=" * 70)

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"'{vocab_path}' not found. Run `python tokenizer.py` first to build the shared "
            f"multilingual character vocabulary before training."
        )
    tokenizer = CharacterTokenizer.load(str(vocab_path))
    logger.info("Loaded tokenizer: vocab_size=%d, pad_id=%d", tokenizer.vocab_size, tokenizer.pad_id)

    feature_extractor = build_feature_extractor()

    logger.info("Collecting training samples (streams from the HF Hub - this can take a while)...")
    train_samples = collect_samples(
        args.languages, split="train",
        max_per_lang=args.max_train_samples_per_lang,
        max_clip_seconds=args.max_clip_seconds,
    )
    logger.info("Collecting dev samples...")
    eval_samples = collect_samples(
        args.languages, split="dev",
        max_per_lang=args.max_eval_samples_per_lang,
        max_clip_seconds=args.max_clip_seconds,
    )

    train_dataset = MultilingualASRDataset(train_samples, tokenizer, feature_extractor)
    eval_dataset = MultilingualASRDataset(eval_samples, tokenizer, feature_extractor)
    logger.info("train_dataset size=%d, eval_dataset size=%d", len(train_dataset), len(eval_dataset))

    collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor)

    model = build_model(args.model_type, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_id)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model built (%s). Total params: %s (%s the 1B hard cap)",
        args.model_type, f"{total_params:,}",
        "under" if total_params < 1_000_000_000 else "OVER",
    )

    training_args = build_training_arguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_fn(tokenizer),
        callbacks=[MemoryMonitorCallback(ram_limit_mb=args.ram_limit_mb)],
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Training complete. Running final evaluation on the dev split...")
    final_metrics = trainer.evaluate()
    logger.info("Final pooled eval metrics: %s", final_metrics)

    per_language_wer: Dict[str, float] = {}
    for lang in args.languages:
        language_samples = [sample for sample in eval_samples if sample.get("lang_code") == lang]
        if not language_samples:
            logger.warning("No eval samples found for language '%s'; skipping per-language WER summary.", lang)
            continue
        language_dataset = MultilingualASRDataset(language_samples, tokenizer, feature_extractor)
        language_metrics = trainer.evaluate(
            eval_dataset=language_dataset,
            metric_key_prefix=f"eval_{lang}",
        )
        lang_wer = language_metrics.get(f"eval_{lang}_wer")
        if lang_wer is None:
            logger.warning("No WER metric returned for language '%s': %s", lang, language_metrics)
            continue
        per_language_wer[lang] = float(lang_wer)

    if per_language_wer:
        overall_wer = compute_unweighted_multilingual_wer(per_language_wer)
        logger.info("Final multilingual score (unweighted mean WER): %.4f", overall_wer)

    best_dir = Path(args.output_dir) / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save(str(best_dir / "vocab.json"))
    logger.info("Saved best model + matching tokenizer to %s", best_dir)


if __name__ == "__main__":
    main()