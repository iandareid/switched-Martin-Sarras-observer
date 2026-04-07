"""Frame adapters between MuJoCo world/body and NED/FRD controller frames."""

from __future__ import annotations

import numpy as np


# MuJoCo world: x-right, y-forward (scene-dependent), z-up.
# Controller world: NED (north-east-down). We map x->north, y->east, z_up->-down.
R_NED_FROM_MJ = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def mj_world_to_ned(v_world: np.ndarray) -> np.ndarray:
    return v_world @ R_NED_FROM_MJ.T


def ned_to_mj_world(v_ned: np.ndarray) -> np.ndarray:
    return v_ned @ R_NED_FROM_MJ


def mj_body_to_frd(v_body: np.ndarray) -> np.ndarray:
    # MuJoCo body frame is treated as FLU; FRD flips Y and Z.
    return v_body * np.array([1.0, -1.0, -1.0], dtype=np.float64)


def frd_to_mj_body(v_frd: np.ndarray) -> np.ndarray:
    return v_frd * np.array([1.0, -1.0, -1.0], dtype=np.float64)


def yaw_ned_to_mj(yaw_ned: np.ndarray) -> np.ndarray:
    # Sign flip due to down-vs-up yaw convention.
    return -yaw_ned

