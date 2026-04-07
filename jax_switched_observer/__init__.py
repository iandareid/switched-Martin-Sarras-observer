from .backend import USING_JAX
from .observer import (
    MODE_BVO,
    MODE_GO,
    MODE_PAE,
    default_config,
    hard_mode_from_logits,
    init_state,
    mode_to_one_hot,
    scan,
    select_mode,
    step,
)
from .types import ObserverConfig, ObserverOutput, ObserverState


def run_parity_demo(*args, **kwargs):
    from .parity_demo import run_parity_demo as _run_parity_demo

    return _run_parity_demo(*args, **kwargs)

__all__ = [
    "MODE_BVO",
    "MODE_GO",
    "MODE_PAE",
    "ObserverConfig",
    "ObserverOutput",
    "ObserverState",
    "USING_JAX",
    "default_config",
    "hard_mode_from_logits",
    "init_state",
    "mode_to_one_hot",
    "run_parity_demo",
    "scan",
    "select_mode",
    "step",
]
