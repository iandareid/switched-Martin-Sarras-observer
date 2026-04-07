from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def single_trajectory_split() -> DatasetSplits:
    idx = np.array([0], dtype=np.int64)
    return DatasetSplits(train=idx, val=idx, test=idx)

