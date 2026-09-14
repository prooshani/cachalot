from __future__ import annotations

import math

import mlx.core as mx


def rms_norm(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """
    Matches DeepSeek RMSNorm:

        dtype = x.dtype
        h = x.float()
        var = mean(h^2, -1, keepdim=True)
        h *= rsqrt(var + eps)
        return (weight * h).to(dtype)
    """
    dtype = x.dtype

    h = x.astype(mx.float32)

    var = mx.mean(
        h * h,
        axis=-1,
        keepdims=True,
    )

    h = h * mx.rsqrt(
        var + eps
    )

    return (
        weight.astype(mx.float32)
        * h
    ).astype(dtype)


def precompute_freqs(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> tuple[mx.array, mx.array]:
    """
    Real-valued equivalent of DeepSeek precompute_freqs_cis().

    Returns:
        cos: [seqlen, dim/2] FP32
        sin: [seqlen, dim/2] FP32
    """
    if dim % 2 != 0:
        raise ValueError(
            f"RoPE dim must be even, got {dim}"
        )

    pair_idx = mx.arange(
        0,
        dim,
        2,
        dtype=mx.float32,
    )

    freqs = 1.0 / mx.power(
        mx.array(base, dtype=mx.float32),
        pair_idx / float(dim),
    )

    if original_seq_len > 0:
        def corrected_dim(
            rotations: float,
        ) -> float:
            return (
                dim
                * math.log(
                    original_seq_len
                    / (
                        rotations
                        * 2.0
                        * math.pi
                    )
                )
                / (
                    2.0
                    * math.log(base)
                )
            )

        low = max(
            math.floor(
                corrected_dim(beta_fast)
            ),
            0,
        )

        high = min(
            math.ceil(
                corrected_dim(beta_slow)
            ),
            dim - 1,
        )

        ramp = (
            mx.arange(
                dim // 2,
                dtype=mx.float32,
            )
            - float(low)
        ) / max(
            float(high - low),
            1e-3,
        )

        ramp = mx.clip(
            ramp,
            0.0,
            1.0,
        )

        smooth = 1.0 - ramp

        freqs = (
            freqs / factor
            * (1.0 - smooth)
            + freqs * smooth
        )

    positions = mx.arange(
        seqlen,
        dtype=mx.float32,
    )

    angles = (
        positions[:, None]
        * freqs[None, :]
    )

    return (
        mx.cos(angles),
        mx.sin(angles),
    )


def apply_rotary_emb(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    *,
    inverse: bool = False,
) -> mx.array:
    """
    Matches DeepSeek's adjacent-pair complex rotation.

    Supported input shapes:
        [S, D]
        [B, S, D]
        [B, S, H, D]

    cos/sin:
        [S, D/2]
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"Last dimension must be even, got {x.shape[-1]}"
        )

    dtype = x.dtype

    xf = x.astype(mx.float32)

    original_shape = xf.shape

    pairs = xf.reshape(
        (*original_shape[:-1], -1, 2)
    )

    real = pairs[..., 0]
    imag = pairs[..., 1]

    if inverse:
        sin = -sin

    if x.ndim == 2:
        # [S, D/2]
        c = cos
        s = sin

    elif x.ndim == 3:
        # [B, S, D/2]
        c = cos[None, :, :]
        s = sin[None, :, :]

    elif x.ndim == 4:
        # [B, S, H, D/2]
        c = cos[
            None,
            :,
            None,
            :,
        ]

        s = sin[
            None,
            :,
            None,
            :,
        ]

    else:
        raise ValueError(
            f"Unsupported RoPE input shape {x.shape}"
        )

    rotated_real = (
        real * c
        - imag * s
    )

    rotated_imag = (
        real * s
        + imag * c
    )

    out = mx.stack(
        [
            rotated_real,
            rotated_imag,
        ],
        axis=-1,
    ).reshape(original_shape)

    return out.astype(dtype)
