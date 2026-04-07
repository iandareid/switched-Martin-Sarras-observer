# JAX Switched Observer

This folder is a drop-in three-mode switched observer library intended for RL integration.

## Scope
- observer only
- functional batched API
- hard masked GO / BVO / PAE mode selection
- RK4 update to match the current C++ observer semantics

## Public API
- `init_state(...)`
- `step(...)`
- `scan(...)`
- `mode_to_one_hot(...)`
- `hard_mode_from_logits(...)`
- `run_parity_demo(...)`

## RL Integration
The intended RL hook is a 3-way mode selector:
- index `0`: GO
- index `1`: BVO
- index `2`: PAE

For direct hard control, pass one-hot mode weights to `step` or `scan`.
If your policy emits logits, convert them first with `hard_mode_from_logits(...)`.

## Parity Replay
The fastest sanity check is to replay the existing C++ three-mode CSV:

```bash
python3 -m jax_switched_observer.parity_demo \
  --reference-csv results/three_mode/switched_demo_three_mode.csv \
  --output-csv results/jax_three_mode/switched_demo_three_mode_jax.csv

python3 tools/plot_switched_demo.py \
  --csv results/jax_three_mode/switched_demo_three_mode_jax.csv \
  --outdir results/jax_three_mode/figures
```

## Dependency Note
Actual GPU execution requires `jax` to be installed in the target repo environment.

This repo environment does not currently provide `jax`, so the library includes a small NumPy-compatible fallback path for smoke testing and CSV parity replay. The functional API is the same either way.
