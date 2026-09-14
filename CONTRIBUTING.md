# Contributing to Cachalot

Thanks for your interest. Cachalot is a performance-sensitive inference runtime, so
contributions follow a measure-first discipline.

## Ground rules

1. **Correctness before speed.** A change that alters model output is not an
   optimization; it is a semantic change and needs explicit discussion.
2. **One architectural change per pull request.** Do not combine budget,
   eviction policy, worker count, prefetch algorithm, and allocator changes.
3. **Every performance claim ships with its benchmark.** Use the scripts in
   `benchmarks/` and paste the before/after JSON in the PR description. Report
   wall time, SSD bytes, hit rate, and MLX active/cache/peak memory. Worker time
   sums are not wall time; do not subtract them from wall clock.
4. **Keep diagnostics behind flags.** Hot-path instrumentation must cost one
   attribute check when disabled.
5. **Tests.** Unit tests must run without the checkpoint (`pytest -q`). Tests that
   need the real model are marked `@pytest.mark.model` and run with `pytest --model`.

## Development setup

```bash
git clone https://github.com/prooshani/cachalot.git
cd cachalot
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,server]"
pytest -q
ruff check src tests benchmarks
```

## Commit messages

Imperative subject line under 72 characters, a blank line, then the reasoning.
Explain *why*, not just *what*. Reference the benchmark file when a change is
performance-motivated.

## Reporting issues

Include: Mac model and memory, macOS version, MLX version, model path layout,
storage device and interface (USB 3.2, Thunderbolt, internal), the exact command,
and the `cachalot doctor` output once that command exists.
