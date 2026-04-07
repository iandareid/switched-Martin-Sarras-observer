"""Core trajectory data structures and interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def unwrap_yaw(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(yaw)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class TrajectoryBuffer:
    t: np.ndarray
    pos: np.ndarray
    yaw: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    yaw_rate: np.ndarray

    @classmethod
    def from_samples(cls, t: np.ndarray, pos: np.ndarray, yaw: np.ndarray) -> "TrajectoryBuffer":
        t = np.asarray(t, dtype=np.float64)
        pos = np.asarray(pos, dtype=np.float64)
        yaw = unwrap_yaw(np.asarray(yaw, dtype=np.float64))
        if t.ndim != 1:
            raise ValueError("Trajectory time must be a 1D array.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Trajectory timestamps must be strictly increasing.")
        if pos.shape != (t.shape[0], 3):
            raise ValueError("Trajectory positions must have shape (N, 3).")
        if yaw.shape != (t.shape[0],):
            raise ValueError("Trajectory yaw must have shape (N,).")
        vel = np.gradient(pos, t, axis=0, edge_order=2)
        acc = np.gradient(vel, t, axis=0, edge_order=2)
        yaw_rate = np.gradient(yaw, t, edge_order=2)
        # Enforce zero terminal derivatives so trajectory end behaves like a hold.
        vel[-1, :] = 0.0
        acc[-1, :] = 0.0
        yaw_rate[-1] = 0.0
        return cls(t=t, pos=pos, yaw=yaw, vel=vel, acc=acc, yaw_rate=yaw_rate)

    @classmethod
    def from_csv(cls, path: Path) -> "TrajectoryBuffer":
        arr = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
        required = ("t", "x", "y", "z", "yaw")
        for field in required:
            if field not in arr.dtype.names:
                raise ValueError(f"Missing CSV field '{field}', required: {required}")
        t = arr["t"]
        pos = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
        yaw = arr["yaw"]
        return cls.from_samples(t=t, pos=pos, yaw=yaw)

    def sample_batch(self, t_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tq_raw = np.asarray(t_query, dtype=np.float64)
        tq = np.clip(tq_raw, self.t[0], self.t[-1])
        pos = np.stack([np.interp(tq, self.t, self.pos[:, i]) for i in range(3)], axis=1)
        vel = np.stack([np.interp(tq, self.t, self.vel[:, i]) for i in range(3)], axis=1)
        acc = np.stack([np.interp(tq, self.t, self.acc[:, i]) for i in range(3)], axis=1)
        yaw = wrap_angle(np.interp(tq, self.t, self.yaw))
        after_end = tq_raw >= self.t[-1]
        if np.any(after_end):
            yaw_end = wrap_angle(np.array([self.yaw[-1]], dtype=np.float64))[0]
            pos[after_end, :] = self.pos[-1, :]
            vel[after_end, :] = 0.0
            acc[after_end, :] = 0.0
            yaw[after_end] = yaw_end
        return pos, vel, acc, yaw


@dataclass
class BatchedTrajectory:
    t: np.ndarray
    pos: np.ndarray
    yaw: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    yaw_rate: np.ndarray
