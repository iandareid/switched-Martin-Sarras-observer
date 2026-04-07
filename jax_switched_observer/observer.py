from __future__ import annotations

from typing import Dict, Iterable, Tuple

from .backend import USING_JAX, jnp, lax, softmax
from .math_utils import cross, make_reference_basis, reconstruct_rotation, safe_normalize
from .types import ObserverConfig, ObserverOutput, ObserverState


MODE_GO = 0
MODE_BVO = 1
MODE_PAE = 2


def mode_to_one_hot(mode_index):
    mode_index = jnp.asarray(mode_index)
    return jnp.eye(3, dtype=jnp.asarray(1.0).dtype)[mode_index]


def hard_mode_from_logits(logits):
    logits = jnp.asarray(logits)
    return mode_to_one_hot(jnp.argmax(logits, axis=-1))


def select_mode(mode_or_logits, already_one_hot=True):
    if already_one_hot:
        return jnp.asarray(mode_or_logits)
    return hard_mode_from_logits(softmax(jnp.asarray(mode_or_logits), axis=-1))


def default_config(alpha_i, beta_i):
    return ObserverConfig(alpha_i=jnp.asarray(alpha_i), beta_i=jnp.asarray(beta_i))


def init_state(alpha_hat0, beta_hat0, b_hat0, b_alpha_hat0, mode0, config):
    mode_one_hot = mode_to_one_hot(mode0)
    alpha_hat0 = jnp.asarray(alpha_hat0)
    beta_hat0 = jnp.asarray(beta_hat0)
    b_hat0 = jnp.asarray(b_hat0)
    b_alpha_hat0 = jnp.asarray(b_alpha_hat0)
    return ObserverState(
        go_alpha_hat=alpha_hat0,
        go_beta_hat=beta_hat0,
        go_b_hat=b_hat0,
        bvo_alpha_m_hat=alpha_hat0 + b_alpha_hat0,
        bvo_beta_m_hat=beta_hat0,
        bvo_b_hat=b_hat0,
        bvo_b_alpha_hat=b_alpha_hat0,
        pae_g_hat=alpha_hat0,
        pae_m_hat=beta_hat0,
        pae_b_hat=b_hat0,
        b_alpha_frozen=b_alpha_hat0,
        mode_one_hot_prev=mode_one_hot,
    )


def _mask(mode_one_hot, index):
    return mode_one_hot[..., index:index + 1]


def _where(mask, on_true, on_false):
    while mask.ndim < on_true.ndim:
        mask = mask[..., None]
    return jnp.where(mask, on_true, on_false)


def _normalize_if(v, enabled, eps):
    out, _ = safe_normalize(v, eps)
    enabled = jnp.asarray(enabled)
    while enabled.ndim < out.ndim:
        enabled = enabled[..., None]
    return jnp.where(enabled, out, v)


def _go_derivative(alpha_hat, beta_hat, b_hat, b_alpha_hat, omega_m, alpha_m, beta_m, config):
    omega_c = omega_m - b_hat
    alpha_corr = alpha_m - b_alpha_hat
    update_enabled = jnp.asarray(config.go_update_b_alpha)
    b_alpha_gain = jnp.where(
        update_enabled,
        jnp.asarray(config.bvo_m_alpha, dtype=alpha_hat.dtype),
        jnp.asarray(0.0, dtype=alpha_hat.dtype),
    )
    return (
        cross(alpha_hat, omega_c) - config.go_k_alpha * (alpha_hat - alpha_corr),
        cross(beta_hat, omega_c) - config.go_k_beta * (beta_hat - beta_m),
        config.go_l_alpha * cross(alpha_hat, alpha_corr) + config.go_l_beta * cross(beta_hat, beta_m),
        b_alpha_gain * cross(omega_c, alpha_hat - alpha_corr),
    )


def _bvo_derivative(alpha_m_hat, beta_m_hat, b_hat, b_alpha_hat, omega_m, alpha_m, beta_m, config):
    omega_c = omega_m - b_hat
    alpha_hat = alpha_m_hat - b_alpha_hat
    return (
        cross(alpha_hat, omega_c) - config.bvo_k_alpha * (alpha_m_hat - alpha_m),
        cross(beta_m_hat, omega_c) - config.bvo_k_beta * (beta_m_hat - beta_m),
        config.bvo_l_beta * cross(beta_m_hat, beta_m),
        config.bvo_m_alpha * cross(omega_c, alpha_m_hat - alpha_m),
    )


def _pae_derivative(g_hat, m_hat, b_hat, omega_m, beta_m, config):
    omega_c = omega_m - b_hat
    magnetic_gravity_dot = jnp.sum(
        _normalize_if(config.alpha_i, True, config.eps_norm)
        * _normalize_if(config.beta_i, True, config.eps_norm),
        axis=-1,
        keepdims=True,
    )
    return (
        cross(g_hat, omega_c)
        - config.pae_k1 * (jnp.sum(beta_m * g_hat, axis=-1, keepdims=True) - magnetic_gravity_dot) * beta_m
        - config.pae_k2 * (jnp.sum(g_hat * g_hat, axis=-1, keepdims=True) - 1.0) * g_hat,
        cross(m_hat, omega_c) - config.pae_k_m * (m_hat - beta_m),
        config.pae_l_m * cross(m_hat, beta_m),
    )


def _rk4_go(alpha_hat, beta_hat, b_hat, b_alpha_hat, omega_m, alpha_m, beta_m, dt, config):
    k1 = _go_derivative(alpha_hat, beta_hat, b_hat, b_alpha_hat, omega_m, alpha_m, beta_m, config)
    k2 = _go_derivative(
        alpha_hat + 0.5 * dt * k1[0],
        beta_hat + 0.5 * dt * k1[1],
        b_hat + 0.5 * dt * k1[2],
        b_alpha_hat + 0.5 * dt * k1[3],
        omega_m,
        alpha_m,
        beta_m,
        config,
    )
    k3 = _go_derivative(
        alpha_hat + 0.5 * dt * k2[0],
        beta_hat + 0.5 * dt * k2[1],
        b_hat + 0.5 * dt * k2[2],
        b_alpha_hat + 0.5 * dt * k2[3],
        omega_m,
        alpha_m,
        beta_m,
        config,
    )
    k4 = _go_derivative(
        alpha_hat + dt * k3[0],
        beta_hat + dt * k3[1],
        b_hat + dt * k3[2],
        b_alpha_hat + dt * k3[3],
        omega_m,
        alpha_m,
        beta_m,
        config,
    )
    alpha_next = alpha_hat + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    beta_next = beta_hat + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    b_next = b_hat + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
    b_alpha_next = b_alpha_hat + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
    alpha_next = _normalize_if(alpha_next, config.renormalize_states, config.eps_norm)
    beta_next = _normalize_if(beta_next, config.renormalize_states, config.eps_norm)
    return alpha_next, beta_next, b_next, b_alpha_next


def _rk4_bvo(alpha_m_hat, beta_m_hat, b_hat, b_alpha_hat, omega_m, alpha_m, beta_m, dt, config):
    k1 = _bvo_derivative(alpha_m_hat, beta_m_hat, b_hat, b_alpha_hat, omega_m, alpha_m, beta_m, config)
    k2 = _bvo_derivative(
        alpha_m_hat + 0.5 * dt * k1[0],
        beta_m_hat + 0.5 * dt * k1[1],
        b_hat + 0.5 * dt * k1[2],
        b_alpha_hat + 0.5 * dt * k1[3],
        omega_m,
        alpha_m,
        beta_m,
        config,
    )
    k3 = _bvo_derivative(
        alpha_m_hat + 0.5 * dt * k2[0],
        beta_m_hat + 0.5 * dt * k2[1],
        b_hat + 0.5 * dt * k2[2],
        b_alpha_hat + 0.5 * dt * k2[3],
        omega_m,
        alpha_m,
        beta_m,
        config,
    )
    k4 = _bvo_derivative(
        alpha_m_hat + dt * k3[0],
        beta_m_hat + dt * k3[1],
        b_hat + dt * k3[2],
        b_alpha_hat + dt * k3[3],
        omega_m,
        alpha_m,
        beta_m,
        config,
    )
    alpha_next = alpha_m_hat + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    beta_next = beta_m_hat + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    b_next = b_hat + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
    b_alpha_next = b_alpha_hat + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
    beta_next = _normalize_if(beta_next, config.renormalize_states, config.eps_norm)
    return alpha_next, beta_next, b_next, b_alpha_next


def _rk4_pae(g_hat, m_hat, b_hat, omega_m, beta_m, dt, config):
    k1 = _pae_derivative(g_hat, m_hat, b_hat, omega_m, beta_m, config)
    k2 = _pae_derivative(
        g_hat + 0.5 * dt * k1[0],
        m_hat + 0.5 * dt * k1[1],
        b_hat + 0.5 * dt * k1[2],
        omega_m,
        beta_m,
        config,
    )
    k3 = _pae_derivative(
        g_hat + 0.5 * dt * k2[0],
        m_hat + 0.5 * dt * k2[1],
        b_hat + 0.5 * dt * k2[2],
        omega_m,
        beta_m,
        config,
    )
    k4 = _pae_derivative(
        g_hat + dt * k3[0],
        m_hat + dt * k3[1],
        b_hat + dt * k3[2],
        omega_m,
        beta_m,
        config,
    )
    g_next = g_hat + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    m_next = m_hat + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    b_next = b_hat + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
    g_next = _normalize_if(g_next, config.renormalize_states, config.eps_norm)
    m_next = _normalize_if(m_next, config.renormalize_states, config.eps_norm)
    return g_next, m_next, b_next


def _active_from_mode(mode_one_hot, go_value, bvo_value, pae_value):
    return (
        _mask(mode_one_hot, MODE_GO) * go_value
        + _mask(mode_one_hot, MODE_BVO) * bvo_value
        + _mask(mode_one_hot, MODE_PAE) * pae_value
    )


def step(state: ObserverState, obs: Dict[str, object], mode_one_hot, dt, config: ObserverConfig) -> Tuple[ObserverState, ObserverOutput]:
    mode_next = select_mode(mode_one_hot, already_one_hot=True)
    mode_prev = state.mode_one_hot_prev

    alpha_m = jnp.asarray(obs["alpha_m"])
    beta_m = jnp.asarray(obs["beta_m"])
    omega_m = jnp.asarray(obs["omega_m"])

    alpha_m = _normalize_if(alpha_m, config.normalize_measurements, config.eps_norm)
    beta_m = _normalize_if(beta_m, config.normalize_measurements | config.normalize_pae_measurement, config.eps_norm)

    prev_go = _mask(mode_prev, MODE_GO)
    prev_bvo = _mask(mode_prev, MODE_BVO)
    prev_pae = _mask(mode_prev, MODE_PAE)
    next_go = _mask(mode_next, MODE_GO)
    next_bvo = _mask(mode_next, MODE_BVO)
    next_pae = _mask(mode_next, MODE_PAE)

    bvo_alpha_hat_prev = state.bvo_alpha_m_hat - state.bvo_b_alpha_hat

    frozen_pre = jnp.where((prev_bvo > 0.5) & (next_bvo < 0.5), state.bvo_b_alpha_hat, state.b_alpha_frozen)

    go_alpha_base = _active_from_mode(mode_prev, state.go_alpha_hat, bvo_alpha_hat_prev, state.pae_g_hat)
    go_beta_base = _active_from_mode(mode_prev, state.go_beta_hat, state.bvo_beta_m_hat, state.pae_m_hat)
    go_b_base = _active_from_mode(mode_prev, state.go_b_hat, state.bvo_b_hat, state.pae_b_hat)

    bvo_alpha_m_base = _active_from_mode(
        mode_prev,
        state.go_alpha_hat + state.b_alpha_frozen,
        state.bvo_alpha_m_hat,
        state.pae_g_hat + state.b_alpha_frozen,
    )
    bvo_beta_base = _active_from_mode(mode_prev, state.go_beta_hat, state.bvo_beta_m_hat, state.pae_m_hat)
    bvo_b_base = _active_from_mode(mode_prev, state.go_b_hat, state.bvo_b_hat, state.pae_b_hat)
    bvo_b_alpha_base = _active_from_mode(
        mode_prev, state.b_alpha_frozen, state.bvo_b_alpha_hat, state.b_alpha_frozen
    )

    pae_g_base = _active_from_mode(mode_prev, state.go_alpha_hat, bvo_alpha_hat_prev, state.pae_g_hat)
    pae_m_base = _active_from_mode(mode_prev, state.go_beta_hat, state.bvo_beta_m_hat, state.pae_m_hat)
    pae_b_base = _active_from_mode(mode_prev, state.go_b_hat, state.bvo_b_hat, state.pae_b_hat)

    go_alpha_next, go_beta_next, go_b_next, go_b_alpha_next = _rk4_go(
        go_alpha_base, go_beta_base, go_b_base, frozen_pre, omega_m, alpha_m, beta_m, dt, config
    )
    bvo_alpha_m_next, bvo_beta_next, bvo_b_next, bvo_b_alpha_next = _rk4_bvo(
        bvo_alpha_m_base, bvo_beta_base, bvo_b_base, bvo_b_alpha_base, omega_m, alpha_m, beta_m, dt, config
    )
    pae_g_next, pae_m_next, pae_b_next = _rk4_pae(
        pae_g_base, pae_m_base, pae_b_base, omega_m, beta_m, dt, config
    )

    frozen_post = _where(
        next_bvo > 0.5,
        bvo_b_alpha_next,
        _where(next_go > 0.5, go_b_alpha_next, frozen_pre),
    )

    state_next = ObserverState(
        go_alpha_hat=_where(next_go > 0.5, go_alpha_next, state.go_alpha_hat),
        go_beta_hat=_where(next_go > 0.5, go_beta_next, state.go_beta_hat),
        go_b_hat=_where(next_go > 0.5, go_b_next, state.go_b_hat),
        bvo_alpha_m_hat=_where(next_bvo > 0.5, bvo_alpha_m_next, state.bvo_alpha_m_hat),
        bvo_beta_m_hat=_where(next_bvo > 0.5, bvo_beta_next, state.bvo_beta_m_hat),
        bvo_b_hat=_where(next_bvo > 0.5, bvo_b_next, state.bvo_b_hat),
        bvo_b_alpha_hat=_where(next_bvo > 0.5, bvo_b_alpha_next, state.bvo_b_alpha_hat),
        pae_g_hat=_where(next_pae > 0.5, pae_g_next, state.pae_g_hat),
        pae_m_hat=_where(next_pae > 0.5, pae_m_next, state.pae_m_hat),
        pae_b_hat=_where(next_pae > 0.5, pae_b_next, state.pae_b_hat),
        b_alpha_frozen=frozen_post,
        mode_one_hot_prev=mode_next,
    )

    alpha_hat = _active_from_mode(mode_next, go_alpha_next, bvo_alpha_m_next - bvo_b_alpha_next, pae_g_next)
    beta_hat = _active_from_mode(mode_next, go_beta_next, bvo_beta_next, pae_m_next)
    b_hat = _active_from_mode(mode_next, go_b_next, bvo_b_next, pae_b_next)
    b_alpha_hat = _active_from_mode(mode_next, frozen_post, bvo_b_alpha_next, frozen_post)
    g_hat_sw = alpha_hat
    m_hat_sw = beta_hat

    reference_basis = make_reference_basis(config.alpha_i, config.beta_i, config.eps_collinear, config.eps_norm)
    r_hat, valid_rotation = reconstruct_rotation(
        alpha_hat, beta_hat, reference_basis, config.eps_collinear, config.eps_norm
    )

    output = ObserverOutput(
        mode_one_hot=mode_next,
        mode_index=jnp.argmax(mode_next, axis=-1),
        switch_flag=jnp.any(mode_next != mode_prev, axis=-1),
        alpha_hat=alpha_hat,
        beta_hat=beta_hat,
        b_hat=b_hat,
        b_alpha_hat=b_alpha_hat,
        g_hat_sw=g_hat_sw,
        m_hat_sw=m_hat_sw,
        r_hat=r_hat,
        valid_rotation=valid_rotation,
    )
    return state_next, output


def _stack_outputs(outputs: Iterable[ObserverOutput]):
    first = outputs[0]
    return ObserverOutput(
        mode_one_hot=jnp.stack([o.mode_one_hot for o in outputs], axis=0),
        mode_index=jnp.stack([o.mode_index for o in outputs], axis=0),
        switch_flag=jnp.stack([o.switch_flag for o in outputs], axis=0),
        alpha_hat=jnp.stack([o.alpha_hat for o in outputs], axis=0),
        beta_hat=jnp.stack([o.beta_hat for o in outputs], axis=0),
        b_hat=jnp.stack([o.b_hat for o in outputs], axis=0),
        b_alpha_hat=jnp.stack([o.b_alpha_hat for o in outputs], axis=0),
        g_hat_sw=jnp.stack([o.g_hat_sw for o in outputs], axis=0),
        m_hat_sw=jnp.stack([o.m_hat_sw for o in outputs], axis=0),
        r_hat=jnp.stack([o.r_hat for o in outputs], axis=0),
        valid_rotation=jnp.stack([o.valid_rotation for o in outputs], axis=0),
    )


def scan(state0: ObserverState, obs_seq: Dict[str, object], mode_seq, dt, config: ObserverConfig):
    keys = ("omega_m", "alpha_m", "beta_m")
    xs = tuple(jnp.asarray(obs_seq[key]) for key in keys) + (jnp.asarray(mode_seq),)

    def body(carry, item):
        omega_t, alpha_t, beta_t, mode_t = item
        next_state, out = step(carry, {"omega_m": omega_t, "alpha_m": alpha_t, "beta_m": beta_t}, mode_t, dt, config)
        return next_state, out

    if USING_JAX:
        return lax.scan(body, state0, xs)

    carry = state0
    outputs = []
    length = xs[0].shape[0]
    for i in range(length):
        carry, out = body(carry, tuple(x[i] for x in xs))
        outputs.append(out)
    return carry, _stack_outputs(outputs)
