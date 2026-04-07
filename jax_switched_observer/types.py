from __future__ import annotations

from typing import NamedTuple


class ObserverConfig(NamedTuple):
    alpha_i: object
    beta_i: object
    go_k_alpha: float = 4.0
    go_k_beta: float = 4.0
    go_l_alpha: float = 1.0
    go_l_beta: float = 1.0
    bvo_k_alpha: float = 4.0
    bvo_k_beta: float = 4.0
    bvo_l_beta: float = 2.0
    bvo_m_alpha: float = 2.0
    go_update_b_alpha: bool = False
    pae_k_m: float = 4.0
    pae_l_m: float = 1.0
    pae_k1: float = 2.0
    pae_k2: float = 2.0
    eps_norm: float = 1e-12
    eps_collinear: float = 1e-9
    normalize_measurements: bool = True
    renormalize_states: bool = True
    normalize_pae_measurement: bool = True


class ObserverState(NamedTuple):
    go_alpha_hat: object
    go_beta_hat: object
    go_b_hat: object
    bvo_alpha_m_hat: object
    bvo_beta_m_hat: object
    bvo_b_hat: object
    bvo_b_alpha_hat: object
    pae_g_hat: object
    pae_m_hat: object
    pae_b_hat: object
    b_alpha_frozen: object
    mode_one_hot_prev: object


class ObserverOutput(NamedTuple):
    mode_one_hot: object
    mode_index: object
    switch_flag: object
    alpha_hat: object
    beta_hat: object
    b_hat: object
    b_alpha_hat: object
    g_hat_sw: object
    m_hat_sw: object
    r_hat: object
    valid_rotation: object
