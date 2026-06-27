from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import psutil
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

DEFAULT_MAX_CLIP_SECONDS: float = 20.0


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Global seed set to %d (random, numpy, torch).", seed)


def get_process_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, ram_limit_mb: float = 8192.0):
        self.ram_limit_mb = ram_limit_mb
        self._breach_logged = False

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        rss_mb = get_process_rss_mb()
        if logs is not None:
            logs["rss_mb"] = round(rss_mb, 1)

        if rss_mb > self.ram_limit_mb and not self._breach_logged:
            logger.warning(
                "RSS memory (%.1f MB) has EXCEEDED the %.1f MB budget at step %d. "
                "Consider lowering --per_device_train_batch_size, --max_clip_seconds, "
                "or --gradient_accumulation_steps.",
                rss_mb, self.ram_limit_mb, state.global_step,
            )
            self._breach_logged = True
        elif rss_mb <= self.ram_limit_mb and self._breach_logged:
            self._breach_logged = False