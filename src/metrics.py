from __future__ import annotations

import logging
from typing import Any, Dict, List

import jiwer
import numpy as np

from tokenizer import CharacterTokenizer


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

LABEL_PAD_VALUE = -100  # Must match collator.py's LABEL_PAD_VALUE.


def ctc_collapse(token_ids: List[int], blank_id: int) -> List[int]:
    collapsed = []
    previous_id = None
    for token_id in token_ids:
        if token_id != previous_id:
            collapsed.append(token_id)
        previous_id = token_id
    return [tid for tid in collapsed if tid != blank_id]


def decode_predictions(
    logits: np.ndarray,
    tokenizer: CharacterTokenizer,
) -> List[str]:

    predicted_ids = np.argmax(logits, axis=-1)  # [batch_size, time_steps]
    decoded_texts = []
    for sample_ids in predicted_ids:
        collapsed = ctc_collapse(sample_ids.tolist(), blank_id=tokenizer.pad_id)
        decoded_texts.append(tokenizer.decode(collapsed))
    return decoded_texts


def decode_labels(
    label_ids: np.ndarray,
    tokenizer: CharacterTokenizer,
) -> List[str]:

    decoded_texts = []
    for sample_ids in label_ids:
        cleaned_ids = [tid if tid != LABEL_PAD_VALUE else tokenizer.pad_id for tid in sample_ids.tolist()]
        decoded_texts.append(tokenizer.decode(cleaned_ids))
    return decoded_texts


def compute_metrics_fn(tokenizer: CharacterTokenizer):

    def compute_metrics(pred: Any) -> Dict[str, float]:

        logits = pred.predictions
        if isinstance(logits, tuple):  # Some model outputs wrap logits in a tuple.
            logits = logits[0]

        predicted_texts = decode_predictions(logits, tokenizer)
        reference_texts = decode_labels(pred.label_ids, tokenizer)

        non_empty_pairs = [(r, p) for r, p in zip(reference_texts, predicted_texts) if r.strip()]
        if not non_empty_pairs:
            logger.warning("compute_metrics called with an all-empty reference batch - returning wer=1.0")
            return {"wer": 1.0}

        refs, hyps = zip(*non_empty_pairs)
        wer = jiwer.wer(list(refs), list(hyps))
        return {"wer": wer}

    return compute_metrics


def compute_unweighted_multilingual_wer(per_language_wer: Dict[str, float]) -> float:

    if not per_language_wer:
        raise ValueError("per_language_wer is empty - nothing to average.")
    values = list(per_language_wer.values())
    mean_wer = sum(values) / len(values)
    logger.info("Unweighted mean WER across %d languages: %.4f", len(values), mean_wer)
    for lang, wer in per_language_wer.items():
        logger.info("  %s: %.4f", lang, wer)
    return mean_wer


if __name__ == "__main__":
    from tokenizer import build_vocab_from_texts

    sample_texts = ["ng'ano saudia kod iran", "habari yako leo", "nĩ ũndũ mwega"]
    char_vocab = build_vocab_from_texts(sample_texts)
    tok = CharacterTokenizer(char_vocab)

    print("=" * 70)
    print("TEST 1: ctc_collapse correctness")
    raw_ids = [5, 5, 5, tok.pad_id, 7, 7, tok.pad_id, tok.pad_id, 9]
    collapsed = ctc_collapse(raw_ids, blank_id=tok.pad_id)
    print(f"  Raw: {raw_ids}")
    print(f"  Collapsed: {collapsed}")
    assert collapsed == [5, 7, 9], f"Expected [5, 7, 9], got {collapsed}"
    print("  PASS")

    print("=" * 70)
    print("TEST 2: Full decode_predictions + decode_labels round trip")

    class FakePred:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids

    text = "habari yako"
    true_ids = tok.encode(text, lang_code=None, add_bos_eos=False)
    expanded_with_blanks = []
    for tid in true_ids:
        expanded_with_blanks.extend([tid, tid, tok.pad_id])  # repeat + trailing blank

    vocab_size = tok.vocab_size
    fake_logits = np.zeros((1, len(expanded_with_blanks), vocab_size), dtype=np.float32)
    for t, tid in enumerate(expanded_with_blanks):
        fake_logits[0, t, tid] = 100.0  # huge logit so argmax reliably picks this ID

    fake_labels = np.array([true_ids + [LABEL_PAD_VALUE] * 3])  # padded shorter than it needs to be, on purpose

    pred = FakePred(predictions=fake_logits, label_ids=fake_labels)
    metrics_fn = compute_metrics_fn(tok)
    result = metrics_fn(pred)
    print(f"  Original text : {text!r}")
    print(f"  WER result    : {result}")
    assert result["wer"] == 0.0, f"Expected perfect WER=0.0 for an exact reconstruction, got {result}"
    print("  PASS")

    print("=" * 70)
    print("TEST 3: Apostrophe sensitivity (the actual linguistic requirement)")
    refs_decoded = decode_labels(np.array([tok.encode("ng'ano saudia", add_bos_eos=False)]), tok)
    print(f"  Decoded reference: {refs_decoded}")
    assert "'" in refs_decoded[0], "Apostrophe must survive the encode/decode round trip"
    print("  PASS - apostrophe preserved through full decode pipeline")

    print("=" * 70)
    print("TEST 4: Unweighted multilingual WER averaging")
    fake_per_lang = {"swa": 0.30, "kik": 0.40, "luo": 0.50, "som": 0.35, "kln": 0.60, "mas": 0.55}
    overall = compute_unweighted_multilingual_wer(fake_per_lang)
    expected = sum(fake_per_lang.values()) / len(fake_per_lang)
    assert abs(overall - expected) < 1e-9
    print(f"  Unweighted mean: {overall:.4f}  (matches manual calculation)")
    print("  PASS")

    print("=" * 70)
    print("ALL TESTS PASSED")