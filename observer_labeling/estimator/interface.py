from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from observer_labeling.data.dataset import TrajectoryDataset


class EstimatorNotImplementedError(NotImplementedError):
    """Raised when the estimator integration seam is exercised before implementation."""


@dataclass
class EstimatorState:
    attitude_quat: np.ndarray
    bias: np.ndarray | None = None
    accel_bias: np.ndarray | None = None
    internal_state: object | None = None


class Estimator(Protocol):
    def init(self, traj: TrajectoryDataset, init_idx: int = 0) -> EstimatorState:
        ...

    def step(
        self,
        est_state: EstimatorState,
        gyro: np.ndarray,
        accel: np.ndarray,
        mag: np.ndarray | None,
        cmd_rates: np.ndarray,
        throttle: np.ndarray,
        mode: int,
    ) -> EstimatorState:
        ...


class UnavailableEstimator:
    def init(self, traj: TrajectoryDataset, init_idx: int = 0) -> EstimatorState:
        raise EstimatorNotImplementedError(
            "Estimator integration is not implemented yet. "
            "Move the estimator into observer_labeling/estimator and wire it here."
        )

    def step(
        self,
        est_state: EstimatorState,
        gyro: np.ndarray,
        accel: np.ndarray,
        mag: np.ndarray | None,
        cmd_rates: np.ndarray,
        throttle: np.ndarray,
        mode: int,
    ) -> EstimatorState:
        raise EstimatorNotImplementedError(
            "Estimator integration is not implemented yet. "
            "Move the estimator into observer_labeling/estimator and wire it here."
        )
