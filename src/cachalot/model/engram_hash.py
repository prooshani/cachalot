from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sympy import isprime
from tokenizers import Regex, normalizers


def build_compressed_token_map(
    tokenizer,
) -> tuple[np.ndarray, int]:
    """
    Exact released DeepSeek V4.1 Engram token
    normalization/compression map.

    Returns:
        token_map:
            [tokenizer_size] int64 array mapping
            tokenizer token IDs to compressed Engram IDs.

        compressed_vocab_size:
            Number of unique normalized token keys.
    """
    sentinel = "\ue000"

    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(
                Regex(r"[ \t\r\n]+"),
                " ",
            ),
            normalizers.Replace(
                Regex(r"^ $"),
                sentinel,
            ),
            normalizers.Strip(),
            normalizers.Replace(
                sentinel,
                " ",
            ),
        ]
    )

    backend = tokenizer.backend_tokenizer

    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)

    for token_id in range(len(tokenizer)):
        text = backend.decode(
            [token_id],
            skip_special_tokens=False,
        )

        if "\ufffd" in text:
            key = backend.id_to_token(
                token_id
            )
        else:
            normalized = (
                normalizer.normalize_str(
                    text
                )
            )

            key = (
                normalized
                if normalized
                else text
            )

        new_id = key_to_new.get(key)

        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id

        lookup[token_id] = new_id

    return (
        np.asarray(
            lookup,
            dtype=np.int64,
        ),
        len(key_to_new),
    )


def find_next_prime(
    start: int,
    seen_primes: set[int],
) -> int:
    candidate = start + 1

    while (
        not isprime(candidate)
        or candidate in seen_primes
    ):
        candidate += 1

    return candidate


@dataclass(frozen=True)
class EngramHashConfig:
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    max_ngram_size: int
    vocab_size: int
    n_heads: int
    compressed_vocab_size: int
    pad_token_id: int


def compute_primes(
    config: EngramHashConfig,
) -> np.ndarray:
    seen: set[int] = set()
    layers = []

    for _ in config.layer_ids:
        per_ngram = []

        for _ in range(config.max_ngram_size - 1):
            sizes = []
            current = config.vocab_size - 1

            for _ in range(config.n_heads):
                current = find_next_prime(
                    current,
                    seen,
                )
                seen.add(current)
                sizes.append(current)

            per_ngram.append(sizes)

        layers.append(per_ngram)

    return np.asarray(
        layers,
        dtype=np.int64,
    )


def compute_offsets(
    primes: np.ndarray,
) -> np.ndarray:
    flat = primes.reshape(
        primes.shape[0],
        -1,
    )

    offsets = np.zeros_like(flat)

    if flat.shape[1] > 1:
        offsets[:, 1:] = np.cumsum(
            flat[:, :-1],
            axis=1,
        )

    return offsets


def compute_hash_multipliers(
    config: EngramHashConfig,
) -> np.ndarray:
    max_long = np.iinfo(np.int64).max

    multiplier_bound = max(
        1,
        (
            max_long
            // config.compressed_vocab_size
        )
        // 2,
    )

    rows = []

    for layer_id in config.layer_ids:
        generator = np.random.default_rng(
            10007 * layer_id
        )

        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(config.max_ngram_size,),
            dtype=np.int64,
        )

        rows.append(values * 2 + 1)

    return np.stack(rows)


class EngramHashState:
    DEAD = -1

    def __init__(
        self,
        config: EngramHashConfig,
        token_map: np.ndarray,
    ) -> None:
        self.config = config

        self.token_map = np.asarray(
            token_map,
            dtype=np.int64,
        )

        self.pad_id = int(
            self.token_map[
                config.pad_token_id
            ]
        )

        self.primes = compute_primes(config)

        self.offsets = compute_offsets(
            self.primes
        )

        self.multipliers = (
            compute_hash_multipliers(config)
        )

        self.history: list[int] = []

    def push(
        self,
        token_id: int,
        *,
        alive: bool = True,
    ) -> np.ndarray:
        if alive:
            compressed = int(
                self.token_map[token_id]
            )
        else:
            compressed = self.DEAD

        self.history.append(compressed)

        tokens = []
        blocked = False

        position = len(self.history) - 1

        for shift in range(
            self.config.max_ngram_size
        ):
            source_pos = position - shift

            if source_pos < 0:
                blocked = True
                source = self.pad_id
            else:
                source = self.history[source_pos]

                if source == self.DEAD:
                    blocked = True

            tokens.append(
                self.pad_id
                if blocked
                else source
            )

        tokens = np.asarray(
            tokens,
            dtype=np.int64,
        )

        # [layers, max_ngram]
        products = (
            tokens[None, :]
            * self.multipliers
        )

        rolling = products[:, 0]

        hashes = []

        for i in range(
            1,
            self.config.max_ngram_size,
        ):
            rolling = np.bitwise_xor(
                rolling,
                products[:, i],
            )

            bucket = (
                rolling[:, None]
                % self.primes[:, i - 1, :]
            )

            hashes.append(bucket)

        # [layers, ngram-1, heads]
        result = np.stack(
            hashes,
            axis=1,
        )

        # Official code flattens ngram/head dimensions.
        result = result.reshape(
            len(self.config.layer_ids),
            -1,
        )

        return result + self.offsets
