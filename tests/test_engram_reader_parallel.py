"""
The parallel and serial Engram row paths must return the same bytes.

Decode asks for 24 rows twice a token and every pread used to be issued from
the decode thread, because the pool only took batches of 64 or more. Against a
drive the expert stream is saturating that cost 26-28 ms per token at a 36 GiB
budget. The pool threshold is now 8 (CACHALOT_ENGRAM_PARALLEL_MIN), so decode
goes through it; each worker writes into its own slice of the output buffer, so
the result is meant to be byte-identical to the serial path whatever the
scheduling. This pins that.
"""
from __future__ import annotations

import numpy as np
import pytest

from cachalot.storage.engram_reader import EngramRowReader, EngramTableLayout

HEAD_DIM = 256
ROWS = 300
DATA_START = 128


@pytest.fixture
def table(tmp_path):
    layout = EngramTableLayout(
        layer=1,
        shard=tmp_path / "engram.bin",
        rows=ROWS,
        head_dim=HEAD_DIM,
        data_start=DATA_START,
        weight_relative_start=0,
        scale_relative_start=ROWS * HEAD_DIM,
    )
    rng = np.random.default_rng(1729)
    weights = rng.integers(0, 256, size=(ROWS, layout.weight_row_bytes), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(ROWS, layout.scale_row_bytes), dtype=np.uint8)
    layout.shard.write_bytes(b"\x00" * DATA_START + weights.tobytes() + scales.tobytes())
    return layout, weights, scales


def _read(layout, ids, parallel_min):
    reader = EngramRowReader()
    reader.PARALLEL_MIN = parallel_min
    try:
        return reader.read_rows(layout, ids)
    finally:
        reader.close()


@pytest.mark.parametrize("count", [1, 7, 8, 24, 63, 64, 120])
def test_parallel_matches_serial_and_the_table(table, count):
    layout, weights, scales = table
    ids = np.random.default_rng(count).integers(0, ROWS, size=count, dtype=np.int64)

    serial = _read(layout, ids, parallel_min=10**9)
    parallel = _read(layout, ids, parallel_min=8)

    assert parallel.weights == serial.weights
    assert parallel.scales == serial.scales
    # And both are the rows the table holds, in the order they were asked for.
    assert parallel.weights == weights[ids].tobytes()
    assert parallel.scales == scales[ids].tobytes()


def test_repeated_ids_keep_their_positions(table):
    layout, weights, scales = table
    ids = np.array([5, 5, 200, 5, 199, 200] * 6, dtype=np.int64)

    parallel = _read(layout, ids, parallel_min=8)

    assert parallel.weights == weights[ids].tobytes()
    assert parallel.scales == scales[ids].tobytes()


def test_default_threshold_covers_a_decode_batch():
    # A decode token asks for 24 rows per Engram layer; the point of the
    # change is that that batch is above the threshold.
    assert EngramRowReader.PARALLEL_MIN <= 24
