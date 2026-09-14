from __future__ import annotations

from dataclasses import dataclass

from v41runtime.storage.index import ExpertEntry, TensorRange
from v41runtime.storage.store import ExpertPayload


@dataclass(frozen=True)
class TensorBytes:
    name: str
    data: memoryview
    size: int


def _find_tensor_bytes(
    tensor: TensorRange,
    payload: ExpertPayload,
) -> TensorBytes:
    for chunk in payload.chunks:
        chunk_end = chunk.start + len(chunk.data)

        if chunk.start <= tensor.start and tensor.end <= chunk_end:
            relative_start = tensor.start - chunk.start
            relative_end = relative_start + tensor.size

            return TensorBytes(
                name=tensor.name,
                data=memoryview(chunk.data)[relative_start:relative_end],
                size=tensor.size,
            )

    raise KeyError(
        f"Tensor {tensor.name} was not found inside expert payload"
    )


def extract_expert_tensors(
    entry: ExpertEntry,
    payload: ExpertPayload,
) -> dict[str, TensorBytes]:
    result: dict[str, TensorBytes] = {}

    for tensor in entry.tensors:
        short_name = tensor.name.rsplit(".", 2)[-2:]
        key = ".".join(short_name)

        result[key] = _find_tensor_bytes(
            tensor,
            payload,
        )

    return result
