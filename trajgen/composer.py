"""Compose multiple segments into trajectory buffers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajgen.types import TrajectoryBuffer


@dataclass
class ComposedSeries:
    t: list[np.ndarray]
    pos: list[np.ndarray]
    yaw: list[np.ndarray]

    def append(self, t: np.ndarray, pos: np.ndarray, yaw: np.ndarray) -> None:
        if t.size == 0:
            return
        self.t.append(t)
        self.pos.append(pos)
        self.yaw.append(yaw)

    def build(self, dt: float) -> TrajectoryBuffer:
        if not self.t:
            raise ValueError("No segments to compose.")
        t = np.concatenate(self.t)
        pos = np.concatenate(self.pos, axis=0)
        yaw = np.concatenate(self.yaw)

        # Append final endpoint sample to preserve segment completion.
        t_final = t[-1] + dt
        t = np.concatenate([t, np.array([t_final], dtype=np.float64)])
        pos = np.concatenate([pos, pos[-1:]], axis=0)
        yaw = np.concatenate([yaw, yaw[-1:]])
        return TrajectoryBuffer.from_samples(t=t, pos=pos, yaw=yaw)


def new_series() -> ComposedSeries:
    return ComposedSeries(t=[], pos=[], yaw=[])

