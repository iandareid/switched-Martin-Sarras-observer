from __future__ import annotations

import os


class GpuUnavailableError(RuntimeError):
    pass


def configure_jax_gpu() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")


def require_gpu_backend() -> tuple[str, list[str]]:
    try:
        import jax

        devices = jax.devices()
        backend = jax.default_backend()
        device_kinds = [device.device_kind for device in devices]
    except Exception as exc:
        raise GpuUnavailableError(f"JAX could not initialize a usable GPU backend: {exc}") from None
    if backend != "gpu" or not any(device.platform == "gpu" for device in devices):
        raise GpuUnavailableError(
            "GPU backend is required for this command, but JAX did not initialize CUDA. "
            f"backend={backend} devices={device_kinds}"
        )
    return backend, device_kinds


def detect_jax_backend() -> tuple[str, list[str]]:
    try:
        import jax

        devices = jax.devices()
        backend = jax.default_backend()
        device_kinds = [device.device_kind for device in devices]
        return backend, device_kinds
    except Exception as exc:
        raise GpuUnavailableError(f"JAX backend detection failed: {exc}") from None
