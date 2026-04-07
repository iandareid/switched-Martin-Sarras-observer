from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import time
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from observer_labeling.data.jax_dataset import JaxTrajectoryDataset
from observer_labeling.env.reward import (
    accel_gravity_alignment,
    beta_pe_matrix,
    quat_angle_error,
    scaled_accel_bias_error,
    scaled_error_norm,
)
from observer_labeling.estimator.jax_impl import (
    JaxEstimatorRuntimeState,
    build_estimator_init_state_jax,
    step_estimator_state_jax,
)
from observer_labeling.estimator.mahony import MahonyParams, build_mahony_init_state_jax, step_mahony_state_jax


MODE_LABELS = ("GO", "BVO", "PAE")
ALL_ACTIONS = (0, 1, 2)
_STATE_KEY_DECIMALS = 5


@dataclass(frozen=True)
class SearchCostConfig:
    w_attitude: float
    w_gyro_bias: float
    w_accel_bias: float
    attitude_pre_break_gain: float
    attitude_break_norm: float
    attitude_transition_width: float
    attitude_post_break_gain: float
    gyro_bias_pre_break_gain: float
    gyro_bias_break_norm: float
    gyro_bias_transition_width: float
    gyro_bias_post_break_gain: float
    accel_bias_pre_break_gain: float
    accel_bias_break_norm: float
    accel_bias_transition_width: float
    accel_bias_post_break_gain: float


@dataclass(frozen=True)
class SearchExecutionConfig:
    hold_steps: int
    estimator_dt: float
    num_decision_steps: int


@dataclass(frozen=True)
class SearchProblem:
    dataset: JaxTrajectoryDataset
    estimator_config: object
    cost_config: SearchCostConfig
    execution_config: SearchExecutionConfig


@dataclass(frozen=True)
class SolverStats:
    nodes_expanded: int
    nodes_pruned: int
    cache_hits: int
    cache_misses: int
    interval_rollouts: int
    observer_updates: int
    max_depth_reached: int
    elapsed_sec: float


@dataclass(frozen=True)
class DecisionSolveResult:
    root_costs: np.ndarray
    best_action: int
    completed: bool
    stats: SolverStats


@dataclass(frozen=True)
class IntervalDiagnostics:
    mag_pe: float
    accel_gravity_angle: float


@dataclass(frozen=True)
class DecisionBoundary:
    decision_index: int
    sample_idx: int
    time_sec: float
    est_state: JaxEstimatorRuntimeState
    interval_diag: IntervalDiagnostics


@dataclass(frozen=True)
class DecisionProfile:
    decision_index: int
    sample_idx: int
    time_sec: float
    depth: int
    best_action: int
    root_costs: np.ndarray
    completed: bool
    nodes_expanded: int
    nodes_pruned: int
    cache_hits: int
    cache_misses: int
    interval_rollouts: int
    observer_updates: int
    elapsed_sec: float
    usec_per_rollout: float
    usec_per_observer_update: float


@dataclass(frozen=True)
class ProjectionSummary:
    num_decisions: int
    mean_decision_sec: float
    median_decision_sec: float
    p95_decision_sec: float
    projected_total_sec_mean: float
    projected_total_sec_p95: float


@dataclass(frozen=True)
class LabeledTrajectoryTrace:
    t: np.ndarray
    true_quat: np.ndarray
    est_quat: np.ndarray
    mahony_quat: np.ndarray
    true_gyro_bias: np.ndarray
    est_gyro_bias: np.ndarray
    mahony_gyro_bias: np.ndarray
    true_accel_bias: np.ndarray
    est_accel_bias: np.ndarray
    attitude_error: np.ndarray
    mahony_attitude_error: np.ndarray
    bias_error: np.ndarray
    bias_error_norm: np.ndarray
    mahony_bias_error: np.ndarray
    mahony_bias_error_norm: np.ndarray
    accel_bias_error: np.ndarray
    accel_bias_error_norm: np.ndarray
    sample_actions: np.ndarray
    decision_t: np.ndarray
    decision_sample_idx: np.ndarray
    root_costs: np.ndarray
    actions: np.ndarray
    search_completed: np.ndarray
    switch_sample_idx: np.ndarray
    switch_times: np.ndarray
    switch_actions: np.ndarray


@dataclass(frozen=True)
class _RolloutResult:
    next_state: JaxEstimatorRuntimeState
    next_sample_idx: int
    interval_cost: float


@dataclass(frozen=True)
class _ChildExpansion:
    action: int
    rollout: _RolloutResult


@dataclass(frozen=True)
class _NodeExpansion:
    ordered_children: tuple[_ChildExpansion, ...]


@dataclass(frozen=True)
class _FixedModeBound:
    cost: float
    completed: bool


@dataclass(frozen=True)
class _ExactSolveResult:
    cost: float
    completed: bool


@dataclass
class _MutableStats:
    nodes_expanded: int = 0
    nodes_pruned: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    interval_rollouts: int = 0
    observer_updates: int = 0
    max_depth_reached: int = 0


@dataclass
class _SearchFrame:
    est_state: JaxEstimatorRuntimeState
    sample_idx: int
    remaining_depth: int
    bound_limit: float
    initialized: bool = False
    node_key: tuple[object, ...] | None = None
    children: tuple[_ChildExpansion, ...] = ()
    child_index: int = 0
    best_cost: float = float("inf")
    completed: bool = True
    pending_step_cost: float | None = None


@dataclass
class _SolverRuntime:
    problem: SearchProblem
    expansion_cache: dict[tuple[object, ...], _NodeExpansion] = field(default_factory=dict)
    exact_cache: dict[tuple[object, ...], _ExactSolveResult] = field(default_factory=dict)
    upper_bound_cache: dict[tuple[object, ...], _FixedModeBound] = field(default_factory=dict)

    def solve(
        self,
        est_state: JaxEstimatorRuntimeState,
        sample_idx: int,
        depth: int,
        node_cap: int | None = None,
        time_cap_sec: float | None = None,
    ) -> DecisionSolveResult:
        start = time.perf_counter()
        stats = _MutableStats()
        root_costs = np.full(len(ALL_ACTIONS), np.inf, dtype=np.float64)
        completed = True
        root_children, root_complete = self._expand_node(
            est_state,
            sample_idx,
            stats,
            node_cap=node_cap,
            time_cap_sec=time_cap_sec,
            start_time=start,
        )
        if not root_complete:
            elapsed_sec = time.perf_counter() - start
            return DecisionSolveResult(
                root_costs=root_costs,
                best_action=0,
                completed=False,
                stats=self._finalize_stats(stats, elapsed_sec),
            )
        for child in root_children.ordered_children:
            if depth <= 1 or child.rollout.next_sample_idx >= (self.problem.dataset.num_samples - 1):
                root_costs[child.action] = child.rollout.interval_cost
                continue
            suffix = self._solve_suffix(
                child.rollout.next_state,
                child.rollout.next_sample_idx,
                depth - 1,
                stats,
                node_cap=node_cap,
                time_cap_sec=time_cap_sec,
                start_time=start,
            )
            root_costs[child.action] = child.rollout.interval_cost + suffix.cost
            completed = completed and suffix.completed
        elapsed_sec = time.perf_counter() - start
        return DecisionSolveResult(
            root_costs=root_costs,
            best_action=int(np.argmin(root_costs)),
            completed=completed,
            stats=self._finalize_stats(stats, elapsed_sec),
        )

    def _finalize_stats(self, stats: _MutableStats, elapsed_sec: float) -> SolverStats:
        return SolverStats(
            nodes_expanded=stats.nodes_expanded,
            nodes_pruned=stats.nodes_pruned,
            cache_hits=stats.cache_hits,
            cache_misses=stats.cache_misses,
            interval_rollouts=stats.interval_rollouts,
            observer_updates=stats.observer_updates,
            max_depth_reached=stats.max_depth_reached,
            elapsed_sec=elapsed_sec,
        )

    def _solve_suffix(
        self,
        est_state: JaxEstimatorRuntimeState,
        sample_idx: int,
        remaining_depth: int,
        stats: _MutableStats,
        node_cap: int | None,
        time_cap_sec: float | None,
        start_time: float,
    ) -> _ExactSolveResult:
        if remaining_depth <= 0 or sample_idx >= (self.problem.dataset.num_samples - 1):
            return _ExactSolveResult(cost=0.0, completed=True)
        root_frame = _SearchFrame(
            est_state=est_state,
            sample_idx=sample_idx,
            remaining_depth=remaining_depth,
            bound_limit=float("inf"),
        )
        stack: list[_SearchFrame] = [root_frame]
        child_result: _ExactSolveResult | None = None
        while stack:
            if time_cap_sec is not None and (time.perf_counter() - start_time) >= time_cap_sec:
                return _ExactSolveResult(
                    cost=child_result.cost if child_result is not None else float("inf"),
                    completed=False,
                )
            frame = stack[-1]
            if child_result is not None and frame.pending_step_cost is not None:
                branch_cost = frame.pending_step_cost + child_result.cost
                if branch_cost < frame.best_cost:
                    frame.best_cost = branch_cost
                frame.completed = frame.completed and child_result.completed
                frame.pending_step_cost = None
                child_result = None
                continue
            if not frame.initialized:
                stats.max_depth_reached = max(stats.max_depth_reached, frame.remaining_depth)
                exact_key = self._exact_key(frame.est_state, frame.sample_idx, frame.remaining_depth)
                cached_exact = self.exact_cache.get(exact_key)
                if cached_exact is not None:
                    stats.cache_hits += 1
                    stack.pop()
                    child_result = cached_exact
                    continue
                stats.cache_misses += 1
                frame.node_key = self._node_key(frame.est_state, frame.sample_idx)
                upper_bound = self._fixed_mode_upper_bound(
                    frame.est_state,
                    frame.sample_idx,
                    frame.remaining_depth,
                    stats,
                    node_cap=node_cap,
                    time_cap_sec=time_cap_sec,
                    start_time=start_time,
                )
                if not np.isfinite(frame.bound_limit):
                    frame.best_cost = upper_bound.cost
                else:
                    frame.best_cost = min(frame.bound_limit, upper_bound.cost)
                frame.completed = upper_bound.completed
                expansion, expansion_complete = self._expand_node(
                    frame.est_state,
                    frame.sample_idx,
                    stats,
                    node_cap=node_cap,
                    time_cap_sec=time_cap_sec,
                    start_time=start_time,
                )
                if not expansion_complete:
                    frame.completed = False
                    stack.pop()
                    child_result = _ExactSolveResult(cost=frame.best_cost, completed=False)
                    continue
                frame.children = expansion.ordered_children
                frame.initialized = True
                continue
            if frame.child_index >= len(frame.children):
                exact_key = self._exact_key(frame.est_state, frame.sample_idx, frame.remaining_depth)
                result = _ExactSolveResult(cost=frame.best_cost, completed=frame.completed)
                if frame.completed:
                    self.exact_cache[exact_key] = result
                stack.pop()
                child_result = result
                continue
            child = frame.children[frame.child_index]
            frame.child_index += 1
            if child.rollout.interval_cost >= frame.best_cost:
                stats.nodes_pruned += 1
                continue
            if child.rollout.next_sample_idx >= (self.problem.dataset.num_samples - 1):
                if child.rollout.interval_cost < frame.best_cost:
                    frame.best_cost = child.rollout.interval_cost
                continue
            frame.pending_step_cost = child.rollout.interval_cost
            stack.append(
                _SearchFrame(
                    est_state=child.rollout.next_state,
                    sample_idx=child.rollout.next_sample_idx,
                    remaining_depth=frame.remaining_depth - 1,
                    bound_limit=frame.best_cost - child.rollout.interval_cost,
                )
            )
        return child_result if child_result is not None else _ExactSolveResult(cost=float("inf"), completed=False)

    def _node_key(self, est_state: JaxEstimatorRuntimeState, sample_idx: int) -> tuple[object, ...]:
        return (int(sample_idx),) + self._quantized_state(est_state)

    def _exact_key(
        self,
        est_state: JaxEstimatorRuntimeState,
        sample_idx: int,
        remaining_depth: int,
    ) -> tuple[object, ...]:
        return (int(remaining_depth),) + self._node_key(est_state, sample_idx)

    def _quantized_state(self, est_state: JaxEstimatorRuntimeState) -> tuple[object, ...]:
        parts: list[object] = []
        for leaf in jax.tree_util.tree_leaves(est_state):
            arr = np.asarray(leaf, dtype=np.float64)
            rounded = np.round(arr, decimals=_STATE_KEY_DECIMALS)
            parts.append(rounded.shape)
            parts.append(rounded.tobytes())
        return tuple(parts)

    def _expand_node(
        self,
        est_state: JaxEstimatorRuntimeState,
        sample_idx: int,
        stats: _MutableStats,
        node_cap: int | None,
        time_cap_sec: float | None,
        start_time: float,
    ) -> tuple[_NodeExpansion, bool]:
        node_key = self._node_key(est_state, sample_idx)
        cached = self.expansion_cache.get(node_key)
        if cached is not None:
            stats.cache_hits += 1
            return cached, True
        stats.cache_misses += 1
        if node_cap is not None and (stats.nodes_expanded + len(ALL_ACTIONS)) > node_cap:
            return _NodeExpansion(tuple()), False
        if time_cap_sec is not None and (time.perf_counter() - start_time) >= time_cap_sec:
            return _NodeExpansion(tuple()), False
        actions = jnp.asarray(ALL_ACTIONS, dtype=jnp.int32)
        next_states, next_indices, costs = _rollout_actions_device(
            est_state=est_state,
            sample_idx=jnp.asarray(sample_idx, dtype=jnp.int32),
            actions=actions,
            dataset=self.problem.dataset,
            estimator_config=self.problem.estimator_config,
            hold_steps=self.problem.execution_config.hold_steps,
            estimator_dt=self.problem.execution_config.estimator_dt,
            cost_config=self.problem.cost_config,
        )
        stats.nodes_expanded += len(ALL_ACTIONS)
        stats.interval_rollouts += len(ALL_ACTIONS)
        stats.observer_updates += int(
            jnp.sum(jnp.maximum(0, next_indices - jnp.asarray(sample_idx, dtype=jnp.int32)))
        )
        children: list[_ChildExpansion] = []
        for idx, action in enumerate(ALL_ACTIONS):
            next_state = jax.tree_util.tree_map(lambda x, i=idx: x[i], next_states)
            rollout = _RolloutResult(
                next_state=_tree_copy(next_state),
                next_sample_idx=int(next_indices[idx]),
                interval_cost=float(costs[idx]),
            )
            children.append(_ChildExpansion(action=int(action), rollout=rollout))
        children.sort(key=lambda child: (child.rollout.interval_cost, child.action))
        expansion = _NodeExpansion(ordered_children=tuple(children))
        self.expansion_cache[node_key] = expansion
        return expansion, True

    def _fixed_mode_upper_bound(
        self,
        est_state: JaxEstimatorRuntimeState,
        sample_idx: int,
        remaining_depth: int,
        stats: _MutableStats,
        node_cap: int | None,
        time_cap_sec: float | None,
        start_time: float,
    ) -> _FixedModeBound:
        exact_key = self._exact_key(est_state, sample_idx, remaining_depth)
        cached = self.upper_bound_cache.get(exact_key)
        if cached is not None:
            stats.cache_hits += 1
            return cached
        stats.cache_misses += 1
        best_cost = float("inf")
        completed = True
        for action in ALL_ACTIONS:
            state = est_state
            idx = sample_idx
            total_cost = 0.0
            for _ in range(remaining_depth):
                if idx >= (self.problem.dataset.num_samples - 1):
                    break
                if node_cap is not None and stats.nodes_expanded >= node_cap:
                    completed = False
                    break
                if time_cap_sec is not None and (time.perf_counter() - start_time) >= time_cap_sec:
                    completed = False
                    break
                rollout = rollout_action(self.problem, state, idx, action)
                total_cost += rollout.interval_cost
                stats.nodes_expanded += 1
                stats.interval_rollouts += 1
                stats.observer_updates += max(0, rollout.next_sample_idx - idx)
                state = rollout.next_state
                idx = rollout.next_sample_idx
            best_cost = min(best_cost, total_cost)
        bound = _FixedModeBound(cost=best_cost, completed=completed)
        self.upper_bound_cache[exact_key] = bound
        return bound


def _tree_copy(tree):
    return jax.tree_util.tree_map(lambda x: x, tree)


def _tree_select(pred: jnp.ndarray, on_true, on_false):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(pred, a, b), on_true, on_false)


def build_search_cost_config(config: dict) -> SearchCostConfig:
    label_search_cfg = config.get("label_search", {})
    return SearchCostConfig(
        w_attitude=float(label_search_cfg["w_attitude"]),
        w_gyro_bias=float(label_search_cfg["w_gyro_bias"]),
        w_accel_bias=float(label_search_cfg["w_accel_bias"]),
        attitude_pre_break_gain=float(label_search_cfg["attitude_pre_break_gain"]),
        attitude_break_norm=float(label_search_cfg["attitude_break_norm"]),
        attitude_transition_width=float(label_search_cfg["attitude_transition_width"]),
        attitude_post_break_gain=float(label_search_cfg["attitude_post_break_gain"]),
        gyro_bias_pre_break_gain=float(label_search_cfg["gyro_bias_pre_break_gain"]),
        gyro_bias_break_norm=float(label_search_cfg["gyro_bias_break_norm"]),
        gyro_bias_transition_width=float(label_search_cfg["gyro_bias_transition_width"]),
        gyro_bias_post_break_gain=float(label_search_cfg["gyro_bias_post_break_gain"]),
        accel_bias_pre_break_gain=float(label_search_cfg["accel_bias_pre_break_gain"]),
        accel_bias_break_norm=float(label_search_cfg["accel_bias_break_norm"]),
        accel_bias_transition_width=float(label_search_cfg["accel_bias_transition_width"]),
        accel_bias_post_break_gain=float(label_search_cfg["accel_bias_post_break_gain"]),
    )


def build_search_execution_config(config: dict, num_samples: int) -> SearchExecutionConfig:
    estimator_rate_hz = float(config["env"]["estimator_rate_hz"])
    hold_steps = int(config["env"]["hold_steps"])
    return SearchExecutionConfig(
        hold_steps=hold_steps,
        estimator_dt=1.0 / estimator_rate_hz,
        num_decision_steps=max((num_samples - 1) // hold_steps, 1),
    )


def build_search_problem(
    dataset: JaxTrajectoryDataset,
    estimator_config,
    cost_config: SearchCostConfig,
    execution_config: SearchExecutionConfig,
) -> SearchProblem:
    return SearchProblem(
        dataset=dataset,
        estimator_config=estimator_config,
        cost_config=cost_config,
        execution_config=execution_config,
    )


def build_search_problem_from_config(
    config: dict,
    dataset: JaxTrajectoryDataset,
    estimator_config_override=None,
) -> tuple[SearchProblem, object]:
    estimator_config, _ = build_estimator_init_state_jax(dataset, init_idx=0)
    if estimator_config_override is not None:
        estimator_config = estimator_config_override
    problem = build_search_problem(
        dataset,
        estimator_config,
        build_search_cost_config(config),
        build_search_execution_config(config, dataset.num_samples),
    )
    return problem, estimator_config


def build_mahony_params_from_config(config: dict, dataset: JaxTrajectoryDataset) -> MahonyParams:
    mahony_cfg = config.get("label_search", {}).get("mahony", {})
    params, _ = build_mahony_init_state_jax(
        dataset,
        init_idx=0,
        k_p=float(mahony_cfg.get("k_p", 2.0)),
        k_i=float(mahony_cfg.get("k_i", 0.1)),
        accel_weight=float(mahony_cfg.get("accel_weight", 1.0)),
        mag_weight=float(mahony_cfg.get("mag_weight", 1.0)),
        accel_gate_margin_mps2=float(mahony_cfg.get("accel_gate_margin_mps2", 0.5)),
        g_ref_mps2=float(mahony_cfg.get("g_ref_mps2", 9.81)),
    )
    return params


def compute_interval_diagnostics(
    dataset: JaxTrajectoryDataset,
    sample_idx: int,
    hold_steps: int,
) -> IntervalDiagnostics:
    max_steps = max(0, min(hold_steps, dataset.num_samples - sample_idx))
    if max_steps <= 0:
        return IntervalDiagnostics(mag_pe=float("nan"), accel_gravity_angle=float("nan"))
    indices = sample_idx + jnp.arange(max_steps, dtype=jnp.int32)
    matrices = jax.vmap(beta_pe_matrix)(dataset.mag_body[indices])
    alignments = jax.vmap(accel_gravity_alignment)(dataset.accel[indices], dataset.true_quat[indices])
    mean_pe = jnp.mean(matrices, axis=0)
    eigvals = jnp.linalg.eigvalsh(mean_pe)
    mean_alignment = jnp.clip(jnp.mean(alignments), -1.0, 1.0)
    return IntervalDiagnostics(
        mag_pe=float(eigvals[0]),
        accel_gravity_angle=float(jnp.arccos(mean_alignment)),
    )


def generate_decision_boundaries(problem: SearchProblem) -> list[DecisionBoundary]:
    _, est_state = build_estimator_init_state_jax(problem.dataset, init_idx=0)
    boundaries: list[DecisionBoundary] = []
    sample_idx = 0
    decision_index = 0
    while sample_idx < (problem.dataset.num_samples - 1):
        boundaries.append(
            DecisionBoundary(
                decision_index=decision_index,
                sample_idx=sample_idx,
                time_sec=float(problem.dataset.t[sample_idx]),
                est_state=_tree_copy(est_state),
                interval_diag=compute_interval_diagnostics(
                    problem.dataset,
                    sample_idx,
                    problem.execution_config.hold_steps,
                ),
            )
        )
        rollout = rollout_action(problem, est_state, sample_idx, 0)
        if rollout.next_sample_idx <= sample_idx:
            break
        est_state = rollout.next_state
        sample_idx = rollout.next_sample_idx
        decision_index += 1
    return boundaries


def rollout_action(
    problem: SearchProblem,
    est_state: JaxEstimatorRuntimeState,
    sample_idx: int,
    action: int,
) -> _RolloutResult:
    next_state, next_sample_idx, total_cost = _rollout_action_device(
        est_state=est_state,
        sample_idx=jnp.asarray(sample_idx, dtype=jnp.int32),
        action=jnp.asarray(action, dtype=jnp.int32),
        dataset=problem.dataset,
        estimator_config=problem.estimator_config,
        hold_steps=problem.execution_config.hold_steps,
        estimator_dt=problem.execution_config.estimator_dt,
        cost_config=problem.cost_config,
    )
    return _RolloutResult(
        next_state=_tree_copy(next_state),
        next_sample_idx=int(next_sample_idx),
        interval_cost=float(total_cost),
    )


@partial(jax.jit, static_argnames=("hold_steps", "estimator_dt", "cost_config"))
def _rollout_action_device(
    est_state: JaxEstimatorRuntimeState,
    sample_idx: jnp.ndarray,
    action: jnp.ndarray,
    dataset: JaxTrajectoryDataset,
    estimator_config,
    hold_steps: int,
    estimator_dt: float,
    cost_config: SearchCostConfig,
):
    def body(carry, _):
        curr_state, curr_idx, total_cost = carry
        valid = curr_idx < dataset.num_samples
        safe_idx = jnp.minimum(curr_idx, dataset.num_samples - 1)
        stepped = step_estimator_state_jax(
            curr_state,
            dataset.gyro[safe_idx],
            dataset.accel[safe_idx],
            dataset.mag_body[safe_idx],
            action,
            estimator_dt,
            estimator_config,
        )
        next_state = _tree_select(valid, stepped, curr_state)
        step_cost = _interval_step_cost(next_state, safe_idx, dataset, cost_config)
        next_total_cost = total_cost + jnp.where(valid, step_cost, 0.0)
        next_idx = curr_idx + jnp.where(valid, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
        return (next_state, next_idx, next_total_cost), None

    (next_state, next_sample_idx, total_cost), _ = jax.lax.scan(
        body,
        (est_state, sample_idx, jnp.asarray(0.0, dtype=jnp.float32)),
        xs=None,
        length=hold_steps,
    )
    return next_state, next_sample_idx, total_cost


@partial(jax.jit, static_argnames=("hold_steps", "estimator_dt", "cost_config"))
def _rollout_actions_device(
    est_state: JaxEstimatorRuntimeState,
    sample_idx: jnp.ndarray,
    actions: jnp.ndarray,
    dataset: JaxTrajectoryDataset,
    estimator_config,
    hold_steps: int,
    estimator_dt: float,
    cost_config: SearchCostConfig,
):
    return jax.vmap(
        _rollout_action_device,
        in_axes=(None, None, 0, None, None, None, None, None),
    )(
        est_state,
        sample_idx,
        actions,
        dataset,
        estimator_config,
        hold_steps,
        estimator_dt,
        cost_config,
    )


def _interval_step_cost(
    next_state: JaxEstimatorRuntimeState,
    safe_idx: jnp.ndarray,
    dataset: JaxTrajectoryDataset,
    cost_config: SearchCostConfig,
) -> jnp.ndarray:
    attitude_error = scaled_error_norm(
        quat_angle_error(dataset.true_quat[safe_idx], next_state.attitude_quat),
        cost_config.attitude_pre_break_gain,
        cost_config.attitude_break_norm,
        cost_config.attitude_transition_width,
        cost_config.attitude_post_break_gain,
    )
    gyro_bias_error = scaled_error_norm(
        jnp.linalg.norm(dataset.true_gyro_bias[safe_idx] - next_state.bias),
        cost_config.gyro_bias_pre_break_gain,
        cost_config.gyro_bias_break_norm,
        cost_config.gyro_bias_transition_width,
        cost_config.gyro_bias_post_break_gain,
    )
    accel_bias_error = scaled_accel_bias_error(
        dataset.true_accel_bias[safe_idx],
        next_state.accel_bias,
        cost_config.accel_bias_pre_break_gain,
        cost_config.accel_bias_break_norm,
        cost_config.accel_bias_transition_width,
        cost_config.accel_bias_post_break_gain,
    )
    return (
        jnp.asarray(cost_config.w_attitude, dtype=jnp.float32) * attitude_error
        + jnp.asarray(cost_config.w_gyro_bias, dtype=jnp.float32) * gyro_bias_error
        + jnp.asarray(cost_config.w_accel_bias, dtype=jnp.float32) * accel_bias_error
    )


def solve_decision(
    problem: SearchProblem,
    est_state: JaxEstimatorRuntimeState,
    sample_idx: int,
    depth: int,
    node_cap: int | None = None,
    time_cap_sec: float | None = None,
) -> DecisionSolveResult:
    runtime = _SolverRuntime(problem)
    return runtime.solve(est_state, sample_idx, depth, node_cap=node_cap, time_cap_sec=time_cap_sec)


def solve_decision_reference(
    problem: SearchProblem,
    est_state: JaxEstimatorRuntimeState,
    sample_idx: int,
    depth: int,
) -> DecisionSolveResult:
    start = time.perf_counter()
    stats = _MutableStats()
    root_costs = np.zeros(len(ALL_ACTIONS), dtype=np.float64)
    for action in ALL_ACTIONS:
        root_costs[action] = _solve_reference_branch(problem, est_state, sample_idx, depth, action, stats)
    elapsed_sec = time.perf_counter() - start
    return DecisionSolveResult(
        root_costs=root_costs,
        best_action=int(np.argmin(root_costs)),
        completed=True,
        stats=SolverStats(
            nodes_expanded=stats.nodes_expanded,
            nodes_pruned=0,
            cache_hits=0,
            cache_misses=0,
            interval_rollouts=stats.interval_rollouts,
            observer_updates=stats.observer_updates,
            max_depth_reached=depth,
            elapsed_sec=elapsed_sec,
        ),
    )


def _solve_reference_branch(
    problem: SearchProblem,
    est_state: JaxEstimatorRuntimeState,
    sample_idx: int,
    depth: int,
    action: int,
    stats: _MutableStats,
) -> float:
    rollout = rollout_action(problem, est_state, sample_idx, action)
    stats.nodes_expanded += 1
    stats.interval_rollouts += 1
    stats.observer_updates += max(0, rollout.next_sample_idx - sample_idx)
    if depth <= 1 or rollout.next_sample_idx >= (problem.dataset.num_samples - 1):
        return rollout.interval_cost
    suffix = min(
        _solve_reference_branch(problem, rollout.next_state, rollout.next_sample_idx, depth - 1, next_action, stats)
        for next_action in ALL_ACTIONS
    )
    return rollout.interval_cost + suffix


def profile_solver(
    problem: SearchProblem,
    boundaries: Sequence[DecisionBoundary],
    depth: int,
    node_cap: int | None = None,
    time_cap_sec: float | None = None,
) -> tuple[list[DecisionProfile], ProjectionSummary]:
    profiles: list[DecisionProfile] = []
    for boundary in boundaries:
        result = solve_decision(
            problem,
            boundary.est_state,
            boundary.sample_idx,
            depth,
            node_cap=node_cap,
            time_cap_sec=time_cap_sec,
        )
        usec_per_rollout = float("nan")
        usec_per_observer_update = float("nan")
        if result.stats.interval_rollouts > 0:
            usec_per_rollout = 1e6 * result.stats.elapsed_sec / float(result.stats.interval_rollouts)
        if result.stats.observer_updates > 0:
            usec_per_observer_update = 1e6 * result.stats.elapsed_sec / float(result.stats.observer_updates)
        profiles.append(
            DecisionProfile(
                decision_index=boundary.decision_index,
                sample_idx=boundary.sample_idx,
                time_sec=boundary.time_sec,
                depth=depth,
                best_action=result.best_action,
                root_costs=result.root_costs,
                completed=result.completed,
                nodes_expanded=result.stats.nodes_expanded,
                nodes_pruned=result.stats.nodes_pruned,
                cache_hits=result.stats.cache_hits,
                cache_misses=result.stats.cache_misses,
                interval_rollouts=result.stats.interval_rollouts,
                observer_updates=result.stats.observer_updates,
                elapsed_sec=result.stats.elapsed_sec,
                usec_per_rollout=usec_per_rollout,
                usec_per_observer_update=usec_per_observer_update,
            )
        )
    elapsed = np.asarray([profile.elapsed_sec for profile in profiles], dtype=np.float64)
    if elapsed.size == 0:
        projection = ProjectionSummary(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    else:
        projection = ProjectionSummary(
            num_decisions=len(profiles),
            mean_decision_sec=float(np.mean(elapsed)),
            median_decision_sec=float(np.median(elapsed)),
            p95_decision_sec=float(np.percentile(elapsed, 95)),
            projected_total_sec_mean=float(np.mean(elapsed) * problem.execution_config.num_decision_steps),
            projected_total_sec_p95=float(np.percentile(elapsed, 95) * problem.execution_config.num_decision_steps),
        )
    return profiles, projection


def label_trajectory(
    problem: SearchProblem,
    depth: int,
    mahony_params: MahonyParams | None = None,
    node_cap: int | None = None,
    time_cap_sec: float | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> LabeledTrajectoryTrace:
    _, est_state = build_estimator_init_state_jax(problem.dataset, init_idx=0)
    if mahony_params is None:
        mahony_params = build_mahony_params_from_config({"label_search": {}}, problem.dataset)
    _, mahony_state = build_mahony_init_state_jax(
        problem.dataset,
        init_idx=0,
        k_p=mahony_params.k_p,
        k_i=mahony_params.k_i,
        accel_weight=mahony_params.accel_weight,
        mag_weight=mahony_params.mag_weight,
        accel_gate_margin_mps2=mahony_params.accel_gate_margin_mps2,
        g_ref_mps2=mahony_params.g_ref_mps2,
    )
    runtime = _SolverRuntime(problem)
    sample_idx = 0
    current_mode = 0

    t_values: list[float] = []
    true_quat_values: list[np.ndarray] = []
    est_quat_values: list[np.ndarray] = []
    mahony_quat_values: list[np.ndarray] = []
    true_gyro_bias_values: list[np.ndarray] = []
    est_gyro_bias_values: list[np.ndarray] = []
    mahony_gyro_bias_values: list[np.ndarray] = []
    true_accel_bias_values: list[np.ndarray] = []
    est_accel_bias_values: list[np.ndarray] = []
    attitude_error_values: list[float] = []
    mahony_attitude_error_values: list[float] = []
    bias_error_values: list[np.ndarray] = []
    bias_error_norm_values: list[float] = []
    mahony_bias_error_values: list[np.ndarray] = []
    mahony_bias_error_norm_values: list[float] = []
    accel_bias_error_values: list[np.ndarray] = []
    accel_bias_error_norm_values: list[float] = []
    sample_action_values: list[int] = []
    decision_t_values: list[float] = []
    decision_sample_idx_values: list[int] = []
    root_cost_values: list[np.ndarray] = []
    action_values: list[int] = []
    search_completed_values: list[int] = []
    switch_sample_idx_values: list[int] = []
    switch_action_values: list[int] = []

    while sample_idx < (problem.dataset.num_samples - 1):
        result = runtime.solve(
            est_state,
            sample_idx,
            depth,
            node_cap=node_cap,
            time_cap_sec=time_cap_sec,
        )
        action = int(result.best_action)
        decision_t_values.append(float(problem.dataset.t[sample_idx]))
        decision_sample_idx_values.append(int(sample_idx))
        root_cost_values.append(np.asarray(result.root_costs, dtype=np.float64))
        action_values.append(action)
        search_completed_values.append(int(result.completed))
        if action != current_mode:
            switch_sample_idx_values.append(int(sample_idx))
            switch_action_values.append(action)
        for _ in range(problem.execution_config.hold_steps):
            if sample_idx >= (problem.dataset.num_samples - 1):
                break
            est_state = step_estimator_state_jax(
                est_state,
                problem.dataset.gyro[sample_idx],
                problem.dataset.accel[sample_idx],
                problem.dataset.mag_body[sample_idx],
                jnp.asarray(action, dtype=jnp.int32),
                problem.execution_config.estimator_dt,
                problem.estimator_config,
            )
            mahony_state = step_mahony_state_jax(
                mahony_state,
                problem.dataset.gyro[sample_idx],
                problem.dataset.accel[sample_idx],
                problem.dataset.mag_body[sample_idx],
                problem.execution_config.estimator_dt,
                mahony_params,
            )
            true_quat = np.asarray(problem.dataset.true_quat[sample_idx], dtype=np.float32)
            est_quat = np.asarray(est_state.attitude_quat, dtype=np.float32)
            mahony_quat = np.asarray(mahony_state.attitude_quat, dtype=np.float32)
            true_gyro_bias = np.asarray(problem.dataset.true_gyro_bias[sample_idx], dtype=np.float32)
            est_gyro_bias = np.asarray(est_state.bias, dtype=np.float32)
            mahony_gyro_bias = np.asarray(mahony_state.gyro_bias, dtype=np.float32)
            true_accel_bias = np.asarray(problem.dataset.true_accel_bias[sample_idx], dtype=np.float32)
            est_accel_bias = np.asarray(est_state.accel_bias, dtype=np.float32)
            gyro_bias_error = true_gyro_bias - est_gyro_bias
            mahony_gyro_bias_error = true_gyro_bias - mahony_gyro_bias
            accel_bias_error = true_accel_bias - est_accel_bias
            t_values.append(float(problem.dataset.t[sample_idx]))
            true_quat_values.append(true_quat)
            est_quat_values.append(est_quat)
            mahony_quat_values.append(mahony_quat)
            true_gyro_bias_values.append(true_gyro_bias)
            est_gyro_bias_values.append(est_gyro_bias)
            mahony_gyro_bias_values.append(mahony_gyro_bias)
            true_accel_bias_values.append(true_accel_bias)
            est_accel_bias_values.append(est_accel_bias)
            attitude_error_values.append(float(quat_angle_error(problem.dataset.true_quat[sample_idx], est_state.attitude_quat)))
            mahony_attitude_error_values.append(
                float(quat_angle_error(problem.dataset.true_quat[sample_idx], mahony_state.attitude_quat))
            )
            bias_error_values.append(gyro_bias_error)
            bias_error_norm_values.append(float(np.linalg.norm(gyro_bias_error)))
            mahony_bias_error_values.append(mahony_gyro_bias_error)
            mahony_bias_error_norm_values.append(float(np.linalg.norm(mahony_gyro_bias_error)))
            accel_bias_error_values.append(accel_bias_error)
            accel_bias_error_norm_values.append(float(np.linalg.norm(accel_bias_error)))
            sample_action_values.append(action)
            sample_idx += 1
        if progress_callback is not None:
            progress_callback(sample_idx / max(problem.dataset.num_samples - 1, 1))
        current_mode = action
    if progress_callback is not None:
        progress_callback(1.0)
    switch_sample_idx = np.asarray(switch_sample_idx_values, dtype=np.int32)
    switch_times = (
        np.asarray(problem.dataset.t[switch_sample_idx], dtype=np.float32)
        if switch_sample_idx.size > 0
        else np.asarray([], dtype=np.float32)
    )
    return LabeledTrajectoryTrace(
        t=np.asarray(t_values, dtype=np.float32),
        true_quat=np.asarray(true_quat_values, dtype=np.float32),
        est_quat=np.asarray(est_quat_values, dtype=np.float32),
        mahony_quat=np.asarray(mahony_quat_values, dtype=np.float32),
        true_gyro_bias=np.asarray(true_gyro_bias_values, dtype=np.float32),
        est_gyro_bias=np.asarray(est_gyro_bias_values, dtype=np.float32),
        mahony_gyro_bias=np.asarray(mahony_gyro_bias_values, dtype=np.float32),
        true_accel_bias=np.asarray(true_accel_bias_values, dtype=np.float32),
        est_accel_bias=np.asarray(est_accel_bias_values, dtype=np.float32),
        attitude_error=np.asarray(attitude_error_values, dtype=np.float32),
        mahony_attitude_error=np.asarray(mahony_attitude_error_values, dtype=np.float32),
        bias_error=np.asarray(bias_error_values, dtype=np.float32),
        bias_error_norm=np.asarray(bias_error_norm_values, dtype=np.float32),
        mahony_bias_error=np.asarray(mahony_bias_error_values, dtype=np.float32),
        mahony_bias_error_norm=np.asarray(mahony_bias_error_norm_values, dtype=np.float32),
        accel_bias_error=np.asarray(accel_bias_error_values, dtype=np.float32),
        accel_bias_error_norm=np.asarray(accel_bias_error_norm_values, dtype=np.float32),
        sample_actions=np.asarray(sample_action_values, dtype=np.int32),
        decision_t=np.asarray(decision_t_values, dtype=np.float32),
        decision_sample_idx=np.asarray(decision_sample_idx_values, dtype=np.int32),
        root_costs=np.asarray(root_cost_values, dtype=np.float64),
        actions=np.asarray(action_values, dtype=np.int32),
        search_completed=np.asarray(search_completed_values, dtype=np.int32),
        switch_sample_idx=switch_sample_idx,
        switch_times=switch_times,
        switch_actions=np.asarray(switch_action_values, dtype=np.int32),
    )


def save_labeled_trajectory(trace: LabeledTrajectoryTrace, path: str | "Path") -> None:
    np.savez(
        path,
        t=trace.t,
        true_quat=trace.true_quat,
        est_quat=trace.est_quat,
        mahony_quat=trace.mahony_quat,
        true_gyro_bias=trace.true_gyro_bias,
        est_gyro_bias=trace.est_gyro_bias,
        mahony_gyro_bias=trace.mahony_gyro_bias,
        true_accel_bias=trace.true_accel_bias,
        est_accel_bias=trace.est_accel_bias,
        attitude_error=trace.attitude_error,
        mahony_attitude_error=trace.mahony_attitude_error,
        bias_error=trace.bias_error,
        bias_error_norm=trace.bias_error_norm,
        mahony_bias_error=trace.mahony_bias_error,
        mahony_bias_error_norm=trace.mahony_bias_error_norm,
        accel_bias_error=trace.accel_bias_error,
        accel_bias_error_norm=trace.accel_bias_error_norm,
        sample_actions=trace.sample_actions,
        decision_t=trace.decision_t,
        decision_sample_idx=trace.decision_sample_idx,
        root_costs=trace.root_costs,
        actions=trace.actions,
        search_completed=trace.search_completed,
        switch_sample_idx=trace.switch_sample_idx,
        switch_times=trace.switch_times,
        switch_actions=trace.switch_actions,
    )


def load_labeled_trajectory(path: str | "Path") -> LabeledTrajectoryTrace:
    data = np.load(path)
    return LabeledTrajectoryTrace(
        t=np.asarray(data["t"]),
        true_quat=np.asarray(data["true_quat"]),
        est_quat=np.asarray(data["est_quat"]),
        mahony_quat=np.asarray(data["mahony_quat"]),
        true_gyro_bias=np.asarray(data["true_gyro_bias"]),
        est_gyro_bias=np.asarray(data["est_gyro_bias"]),
        mahony_gyro_bias=np.asarray(data["mahony_gyro_bias"]),
        true_accel_bias=np.asarray(data["true_accel_bias"]),
        est_accel_bias=np.asarray(data["est_accel_bias"]),
        attitude_error=np.asarray(data["attitude_error"]),
        mahony_attitude_error=np.asarray(data["mahony_attitude_error"]),
        bias_error=np.asarray(data["bias_error"]),
        bias_error_norm=np.asarray(data["bias_error_norm"]),
        mahony_bias_error=np.asarray(data["mahony_bias_error"]),
        mahony_bias_error_norm=np.asarray(data["mahony_bias_error_norm"]),
        accel_bias_error=np.asarray(data["accel_bias_error"]),
        accel_bias_error_norm=np.asarray(data["accel_bias_error_norm"]),
        sample_actions=np.asarray(data["sample_actions"]),
        decision_t=np.asarray(data["decision_t"]),
        decision_sample_idx=np.asarray(data["decision_sample_idx"]),
        root_costs=np.asarray(data["root_costs"]),
        actions=np.asarray(data["actions"]),
        search_completed=np.asarray(data["search_completed"]),
        switch_sample_idx=np.asarray(data["switch_sample_idx"]),
        switch_times=np.asarray(data["switch_times"]),
        switch_actions=np.asarray(data["switch_actions"]),
    )


def iter_depths(min_depth: int, max_depth: int) -> Iterable[int]:
    for depth in range(min_depth, max_depth + 1):
        yield depth
