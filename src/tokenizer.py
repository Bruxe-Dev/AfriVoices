from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS: List[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# One dedicated routing tag per target language, used so a single shared
# multilingual model can be told (or can infer) which language a given
# utterance belongs to.
LANGUAGE_TAGS: Dict[str, str] = {
    "swa": "[SWA]",
    "kik": "[KIK]",
    "luo": "[LUO]",
    "som": "[SOM]",
    "kln": "[KLN]",
    "mas": "[MAS]",
}


def build_vocab_from_texts(texts: Iterable[str]) -> List[str]:

    unique_chars = set()
    for text in texts:
        if not text:
            continue
        unique_chars.update(text)
    return sorted(unique_chars)


class CharacterTokenizer:

    def __init__(self, char_vocab: List[str]):

        all_tokens = list(SPECIAL_TOKENS) + list(LANGUAGE_TAGS.values()) + list(char_vocab)

        seen = set()
        deduped_tokens = []
        for tok in all_tokens:
            if tok not in seen:
                seen.add(tok)
                deduped_tokens.append(tok)

        self.id_to_token: Dict[int, str] = {i: tok for i, tok in enumerate(deduped_tokens)}
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(deduped_tokens)}

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Total number of tokens, including special tokens and language tags."""
        return len(self.id_to_token)

    def lang_tag_id(self, lang_code: str) -> int:

        tag = LANGUAGE_TAGS[lang_code]
        return self.token_to_id[tag]

    def encode(
        self,
        text: str,
        lang_code: Optional[str] = None,
        add_bos_eos: bool = True,
    ) -> List[int]:

        ids: List[int] = []
        if lang_code is not None:
            ids.append(self.lang_tag_id(lang_code))
        if add_bos_eos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.token_to_id.get(ch, self.unk_id))
        if add_bos_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:

        special_set = set(SPECIAL_TOKENS) | set(LANGUAGE_TAGS.values())
        chars = []
        for token_id in ids:
            token = self.id_to_token.get(token_id, UNK_TOKEN)
            if skip_special_tokens and token in special_set:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str) -> None:

        payload = {
            "token_to_id": self.token_to_id,
            "special_tokens": SPECIAL_TOKENS,
            "language_tags": LANGUAGE_TAGS,
            "vocab_size": self.vocab_size,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved vocabulary (%d tokens) to %s", self.vocab_size, path)

    @classmethod
    def load(cls, path: str) -> "CharacterTokenizer":

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        token_to_id = payload["token_to_id"]

        instance = cls(char_vocab=[])
        instance.token_to_id = {tok: int(i) for tok, i in token_to_id.items()}
        instance.id_to_token = {int(i): tok for tok, i in token_to_id.items()}
        instance.pad_id = instance.token_to_id[PAD_TOKEN]
        instance.bos_id = instance.token_to_id[BOS_TOKEN]
        instance.eos_id = instance.token_to_id[EOS_TOKEN]
        instance.unk_id = instance.token_to_id[UNK_TOKEN]
        return instance

    def audit_report(self) -> str:
        """
        Produce a human-readable summary for the Linguistic Lead to verify
        that critical characters got dedicated token IDs, without needing
        to read raw JSON.

        Returns:
            Multi-line string report.
        """
        critical_chars = ["ĩ", "ũ", "'"]
        lines = [f"Vocabulary audit report ({self.vocab_size} total tokens)", "=" * 50]
        lines.append(f"Special tokens: {SPECIAL_TOKENS}")
        lines.append(f"Language tags : {list(LANGUAGE_TAGS.values())}")
        lines.append("-" * 50)
        lines.append("Critical character check:")
        for ch in critical_chars:
            if ch in self.token_to_id:
                lines.append(f"  {ch!r:>6} -> token ID {self.token_to_id[ch]}  [OK: dedicated ID]")
            else:
                lines.append(f"  {ch!r:>6} -> NOT FOUND IN VOCAB  [check training sample coverage]")
        return "\n".join(lines)


if __name__ == "__main__":

    SAMPLES_PER_LANGUAGE = 200

    import kenyan_loader
    import swahili_loader

    all_texts: List[str] = []

    logger.info("Collecting samples from Swahili (health domain, train split)...")
    count = 0
    for sample in swahili_loader.iter_domain_samples("health", "train"):
        all_texts.append(sample["clean_text"])
        count += 1
        if count >= SAMPLES_PER_LANGUAGE:
            break
    logger.info("  Collected %d Swahili samples.", count)

    for lang in kenyan_loader.VALID_LANGS:
        logger.info("Collecting samples from %s...", lang)
        count = 0
        for sample in kenyan_loader.iter_language_samples(lang, split="train"):
            all_texts.append(sample["clean_text"])
            count += 1
            if count >= SAMPLES_PER_LANGUAGE:
                break
        logger.info("  Collected %d %s samples.", count, lang)

    logger.info("Building character vocabulary from %d total text samples...", len(all_texts))
    char_vocab = build_vocab_from_texts(all_texts)
    tokenizer = CharacterTokenizer(char_vocab)

    logger.info("Vocabulary built: %d total tokens (including special/language tags).", tokenizer.vocab_size)
    print()
    print(tokenizer.audit_report())

    tokenizer.save("vocab.json")

    test_text = "nĩ ũndũ mwega ng'ung'unyek"
    encoded = tokenizer.encode(test_text, lang_code="kik")
    decoded = tokenizer.decode(encoded)
    logger.info("Round-trip test: %r -> %d ids -> %r", test_text, len(encoded), decoded)