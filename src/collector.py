from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import Wav2Vec2FeatureExtractor

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

LABEL_PAD_VALUE = -100  # CTC loss ignore-index sentinel - see module docstring.


@dataclass
class DataCollatorCTCWithPadding:
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        for i, f in enumerate(features):
            if "input_values" not in f or "labels" not in f:
                raise KeyError(
                    f"Sample at batch index {i} is missing 'input_values' or 'labels'. "
                    f"Got keys: {list(f.keys())}"
                )

        # --- Audio: delegate to the feature extractor's own padding --- #
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # --- Labels: hand-rolled padding with the CTC ignore sentinel --- #
        label_sequences = [f["labels"] for f in features]
        max_label_length = max(len(seq) for seq in label_sequences)

        padded_labels = torch.full(
            (len(label_sequences), max_label_length),
            fill_value=LABEL_PAD_VALUE,
            dtype=torch.long,
        )
        for i, seq in enumerate(label_sequences):
            padded_labels[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

        batch["labels"] = padded_labels
        return batch


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from model import build_feature_extractor

    fe = build_feature_extractor()
    collator = DataCollatorCTCWithPadding(feature_extractor=fe)

    import numpy as np
    fake_batch = [
        {"input_values": np.random.randn(8000).astype(np.float32), "labels": [1, 5, 6, 2]},
        {"input_values": np.random.randn(16000).astype(np.float32), "labels": [1, 5, 6, 7, 8, 9, 2]},
        {"input_values": np.random.randn(12000).astype(np.float32), "labels": [1, 5, 2]},
    ]

    result = collator(fake_batch)
    print("Batch keys:", list(result.keys()))
    print("input_values shape  :", tuple(result["input_values"].shape))
    print("attention_mask shape:", tuple(result["attention_mask"].shape))
    print("labels shape        :", tuple(result["labels"].shape))
    print("labels tensor:")
    print(result["labels"])

    # Verify padding correctness explicitly, not just "it ran without error".
    assert result["input_values"].shape[1] == 16000, "Should pad to the LONGEST audio in the batch"
    assert result["labels"].shape[1] == 7, "Should pad to the LONGEST label sequence in the batch"
    assert (result["labels"][0, 4:] == LABEL_PAD_VALUE).all(), "Padding positions must be exactly -100"
    assert (result["labels"][2, 3:] == LABEL_PAD_VALUE).all(), "Padding positions must be exactly -100"
    assert (result["labels"][1] != LABEL_PAD_VALUE).all(), "The longest sequence should have NO padding at all"
    print("\nALL ASSERTIONS PASSED")