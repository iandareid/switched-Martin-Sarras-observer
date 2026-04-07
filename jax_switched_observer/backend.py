from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    USING_JAX = True
except ImportError:  # pragma: no cover - exercised indirectly in this environment
    import numpy as jnp

    jax = None
    USING_JAX = False

    class _Lax:
        @staticmethod
        def select(pred, on_true, on_false):
            return jnp.where(pred, on_true, on_false)

        @staticmethod
        def scan(fn, init, xs):
            carry = init
            outputs = []
            length = xs[0].shape[0]
            for i in range(length):
                item = tuple(x[i] for x in xs)
                carry, y = fn(carry, item)
                outputs.append(y)
            return carry, outputs

    lax = _Lax()


def softmax(logits, axis=-1):
    shifted = logits - jnp.max(logits, axis=axis, keepdims=True)
    exp = jnp.exp(shifted)
    return exp / jnp.sum(exp, axis=axis, keepdims=True)
