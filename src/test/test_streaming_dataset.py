import unittest

import numpy as np

from src.train import StreamingMultilingualASRDataset


class DummyTokenizer:
    def encode(self, text, lang_code=None, add_bos_eos=True):
        return [len(text)] if add_bos_eos else [len(text)]


class DummyFeatureExtractor:
    def __call__(self, audio_array, sampling_rate=16000, return_tensors="np"):
        return {"input_values": [np.asarray(audio_array, dtype=np.float32)]}


class StreamingMultilingualASRDatasetTest(unittest.TestCase):
    def test_iterates_samples_without_materializing_all_items(self):
        samples = [
            {
                "clean_text": "hello",
                "lang_code": "eng",
                "audio_array": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "target_sample_rate": 16000,
            },
            {
                "clean_text": "world",
                "lang_code": "eng",
                "audio_array": np.array([0.4, 0.5], dtype=np.float32),
                "target_sample_rate": 16000,
            },
        ]

        dataset = StreamingMultilingualASRDataset(
            sample_iterator=iter(samples),
            tokenizer=DummyTokenizer(),
            feature_extractor=DummyFeatureExtractor(),
        )

        items = list(dataset)

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["labels"], [5])
        self.assertTrue(np.allclose(items[0]["input_values"], np.array([0.1, 0.2, 0.3], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
