from __future__ import annotations

import os
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from cachalot.storage.tensor_index import CheckpointTensor


@dataclass(frozen=True)
class ResidentTensor:
    name: str
    checkpoint_dtype: str
    shape: tuple[int, ...]
    data: mx.array

    @property
    def size(self) -> int:
        return self.data.nbytes


def _read_exact(
    tensor: CheckpointTensor,
) -> bytes:
    fd = os.open(
        tensor.shard,
        os.O_RDONLY,
    )

    try:
        raw = os.pread(
            fd,
            tensor.size,
            tensor.start,
        )
    finally:
        os.close(fd)

    if len(raw) != tensor.size:
        raise OSError(
            f"Short read for {tensor.name}: "
            f"{len(raw)} != {tensor.size}"
        )

    return raw


def load_resident_tensor(
    tensor: CheckpointTensor,
) -> ResidentTensor:
    raw = _read_exact(tensor)

    if tensor.dtype == "F32":
        np_array = np.frombuffer(
            raw,
            dtype=np.float32,
        ).reshape(tensor.shape)

        data = mx.array(np_array)

    elif tensor.dtype == "BF16":
        # NumPy does not have a portable native bfloat16 dtype.
        # Preserve the exact 16-bit BF16 bit pattern first.
        bits = np.frombuffer(
            raw,
            dtype=np.uint16,
        ).reshape(tensor.shape)

        # MLX can reinterpret uint16 bits as BF16.
        data = mx.array(bits).view(mx.bfloat16)

    elif tensor.dtype in {
        "F8_E4M3",
        "F8_E8M0",
        "I8",
    }:
        # Preserve quantized checkpoint bytes exactly.
        np_array = np.frombuffer(
            raw,
            dtype=np.uint8,
        ).reshape(tensor.shape)

        data = mx.array(np_array)

    else:
        raise NotImplementedError(
            f"Unsupported checkpoint dtype "
            f"{tensor.dtype!r} for {tensor.name}"
        )

    mx.eval(data)

    return ResidentTensor(
        name=tensor.name,
        checkpoint_dtype=tensor.dtype,
        shape=tensor.shape,
        data=data,
    )
