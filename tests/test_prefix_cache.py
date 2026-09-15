import mlx.core as mx

from cachalot.model.prefix_cache import PrefixCache, SequenceSnapshot, common_prefix_len


def snap(tokens):
    return SequenceSnapshot(
        tokens=tuple(tokens),
        position=len(tokens),
        logits=mx.zeros((4,)),
        windows={},
        compressed_caches={},
        compressor_kv={},
        compressor_score={},
        indexer_k={},
        engram_history=list(tokens),
        shared_compress_kv=None,
        shared_index_k_layer=None,
        shared_topk_idxs=None,
        shared_candidates=None,
    )


def test_common_prefix():
    assert common_prefix_len((1, 2, 3), (1, 2, 4)) == 2
    assert common_prefix_len((), (1,)) == 0


def test_find_longest_prefix_and_counters():
    pc = PrefixCache(max_entries=3)
    pc.add(snap([1, 2]))
    pc.add(snap([1, 2, 3, 4]))
    pc.add(snap([9, 9]))

    best = pc.find((1, 2, 3, 4, 5, 6))
    assert best is not None and best.tokens == (1, 2, 3, 4)
    assert pc.find((1, 2, 3, 4)).tokens == (1, 2, 3, 4)  # exact repeat qualifies
    assert pc.find((1, 3)) is None
    assert pc.find((7,)) is None
    assert (pc.hits, pc.misses, pc.reused_tokens) == (2, 2, 8)


def test_capacity_and_dedup():
    pc = PrefixCache(max_entries=2)
    pc.add(snap([1]))
    pc.add(snap([1, 2]))
    pc.add(snap([1, 2]))  # same tokens replaces, does not grow
    assert len(pc) == 2
    pc.add(snap([1, 2, 3]))
    assert len(pc) == 2
    assert pc.find((1,)) is None  # oldest evicted


def test_default_capacity_covers_interleaved_conversations():
    pc = PrefixCache()
    convs = [[100 + c] for c in range(6)]
    for turn in range(2):
        for c, tokens in enumerate(convs):
            tokens.append(turn)
            pc.add(snap(tokens))          # after prompt
            tokens.append(50 + c)
            pc.add(snap(tokens))          # after reply
    # every conversation's latest state must still be findable
    for tokens in convs:
        assert pc.find(tuple(tokens) + (7, 8)) is not None
