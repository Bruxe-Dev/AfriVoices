import sys
import types
import unittest

import numpy as np


# Minimal stubs so the dataset module can be imported without the full training stack.
torch_stub = types.ModuleType("torch")
torch_utils_stub = types.ModuleType("torch.utils")
torch_data_stub = types.ModuleType("torch.utils.data")


class Dataset:  # noqa: D401 - simple stub for the dataset base class
    pass


torch_data_stub.Dataset = Dataset
torch_utils_stub.data = torch_data_stub
torch_stub.utils = torch_utils_stub
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.utils", torch_utils_stub)
sys.modules.setdefault("torch.utils.data", torch_data_stub)

transformers_stub = types.ModuleType("transformers")


class Trainer:  # noqa: D401 - simple stub for import compatibility
    pass


class Wav2Vec2ForCTC:  # noqa: D401 - simple stub for import compatibility
    pass


transformers_stub.Trainer = Trainer
transformers_stub.Wav2Vec2ForCTC = Wav2Vec2ForCTC
sys.modules.setdefault("transformers", transformers_stub)


class DummyCharacterTokenizer:
    def encode(self, text, lang_code=None, add_bos_eos=True):
        return [1, 2, 3]


# Stub the local training modules that are unrelated to the normalization behavior.
kenyan_loader_stub = types.ModuleType("src.kenyan_loader")
kenyan_loader_stub.VALID_LANGS = ("eng",)
kenyan_loader_stub.iter_language_samples = lambda *args, **kwargs: iter(())
sys.modules["src.kenyan_loader"] = kenyan_loader_stub
sys.modules["kenyan_loader"] = kenyan_loader_stub

swahili_loader_stub = types.ModuleType("src.swahili_loader")
swahili_loader_stub.VALID_DOMAINS = ()
swahili_loader_stub.iter_domain_samples = lambda *args, **kwargs: iter(())
sys.modules["src.swahili_loader"] = swahili_loader_stub
sys.modules["swahili_loader"] = swahili_loader_stub

collector_stub = types.ModuleType("src.collector")


class DataCollatorCTCWithPadding:  # noqa: D401 - simple stub for import compatibility
    def __init__(self, feature_extractor=None):
        self.feature_extractor = feature_extractor


collector_stub.DataCollatorCTCWithPadding = DataCollatorCTCWithPadding
sys.modules["src.collector"] = collector_stub
sys.modules["collector"] = collector_stub

metrics_stub = types.ModuleType("src.metrics")
metrics_stub.compute_metrics_fn = lambda tokenizer: (lambda pred: {"wer": 0.0})
metrics_stub.compute_unweighted_multilingual_wer = lambda per_language_wer: 0.0
sys.modules["src.metrics"] = metrics_stub
sys.modules["metrics"] = metrics_stub

model_stub = types.ModuleType("src.model")
model_stub.build_baseline_config = lambda *args, **kwargs: None
model_stub.build_feature_extractor = lambda: None
model_stub.build_pretrained_model = lambda *args, **kwargs: None
sys.modules["src.model"] = model_stub
sys.modules["model"] = model_stub

tokenizer_stub = types.ModuleType("src.tokenizer")

tokenizer_stub.CharacterTokenizer = DummyCharacterTokenizer
sys.modules["src.tokenizer"] = tokenizer_stub
sys.modules["tokenizer"] = tokenizer_stub

train_config_stub = types.ModuleType("src.train_config")
train_config_stub.build_training_arguments = lambda *args, **kwargs: None
sys.modules["src.train_config"] = train_config_stub
sys.modules["train_config"] = train_config_stub

utils_stub = types.ModuleType("src.utils")
utils_stub.DEFAULT_MAX_CLIP_SECONDS = 20.0
utils_stub.MemoryMonitorCallback = object
utils_stub.get_logger = lambda name: type("Logger", (), {"info": lambda *args, **kwargs: None, "warning": lambda *args, **kwargs: None})()
utils_stub.set_global_seed = lambda seed: None
sys.modules["src.utils"] = utils_stub
sys.modules["utils"] = utils_stub

from src.train import MultilingualASRDataset


class DummyFeatureExtractor:
    def __call__(self, audio_array, sampling_rate=16000, return_tensors=None):
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.size == 0:
            normalized = audio
        else:
            std = np.std(audio)
            if std == 0:
                normalized = audio - np.mean(audio)
            else:
                normalized = (audio - np.mean(audio)) / std
        return {"input_values": np.array([normalized], dtype=np.float32)}


class MultilingualASRDatasetNormalizationTest(unittest.TestCase):
    def test_dataset_applies_feature_extractor_before_returning_audio(self):
        tokenizer = type("DummyTokenizer", (), {"encode": lambda self, text, lang_code=None, add_bos_eos=True: [1, 2, 3]})()
        feature_extractor = DummyFeatureExtractor()
        dataset = MultilingualASRDataset(
            samples=[{"audio_array": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), "clean_text": "hello", "target_sample_rate": 16000}],
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )

        item = dataset[0]

        self.assertIn("input_values", item)
        self.assertIn("labels", item)
        np.testing.assert_allclose(np.mean(item["input_values"]), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.std(item["input_values"]), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
