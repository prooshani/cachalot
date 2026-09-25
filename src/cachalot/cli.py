"""
`cachalot` command line.

    cachalot serve   --model PATH [--host --port --expert-budget-gib --max-seq-len]
    cachalot chat    --model PATH [PROMPT ...]
    cachalot doctor  [--model PATH]
    cachalot bench   --model PATH [--prompt-tokens N --decode-tokens N]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cachalot import __version__
from cachalot.config import GiB, RuntimeConfig, load_config


def _budget(value: str) -> float:
    if str(value).lower() in ("auto", "0"):
        return 0.0
    return float(value)


def _runtime_args(parser: argparse.ArgumentParser, cfg: RuntimeConfig) -> None:
    parser.add_argument("--model", default=cfg.resolved_model_path, help="Path to the DeepSeek-V4.1-Flash checkpoint (default: discovered).")
    parser.add_argument("--max-seq-len", type=int, default=cfg.max_seq_len)
    parser.add_argument(
        "--expert-budget-gib",
        type=_budget,
        default=0.0,
        help=f"Resident routed-expert budget in GiB, or 'auto' (default; {cfg.expert_cache_budget_gib:.0f} GiB on this machine).",
    )
    parser.add_argument("--io-workers", type=int, default=cfg.io_workers)
    parser.add_argument(
        "--family",
        choices=("auto", "deepseek", "glm", "minimax"),
        default=os.environ.get("CACHALOT_MODEL_FAMILY", "auto"),
        help="Model family; auto reads the checkpoint's config.json (glm5_next means GLM-5.3-Flash, minimax_m3 MiniMax-M3).",
    )
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        prog="cachalot",
        description="SSD-backed DeepSeek V4.1 Flash runtime for Apple Silicon.",
    )
    parser.add_argument("--version", action="version", version=f"cachalot {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="Start the OpenAI-compatible HTTP server.")
    _runtime_args(p, cfg)
    p.add_argument("--host", default=cfg.host)
    p.add_argument("--port", type=int, default=cfg.port)
    p.add_argument("--model-id", default=cfg.model_id, help="Name reported by /v1/models.")
    p.add_argument("--api-key", default=os.environ.get("CACHALOT_API_KEY"), help="Require this Bearer token.")
    p.add_argument(
        "--snapshot-dir",
        default=os.environ.get("CACHALOT_SNAPSHOT_DIR"),
        help="Keep system-prompt prefix snapshots here across restarts (default: memory only).",
    )
    p.add_argument("--default-max-tokens", type=int, default=1024)
    p.add_argument("--default-temperature", type=float, default=0.6)
    p.add_argument("--thinking", action="store_true", help="Default requests to thinking mode.")
    p.add_argument("--reasoning-effort", default=None, help="Default reasoning effort (1-100, low, high, max).")

    p = sub.add_parser("chat", help="Chat in the terminal (interactive if no prompt given).")
    _runtime_args(p, cfg)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--frequency-penalty", type=float, default=0.0,
                   help="subtract this per prior occurrence of a token; the control that "
                        "breaks a repetition loop, because it grows with the count")
    p.add_argument("--presence-penalty", type=float, default=0.0,
                   help="subtract this once for any token already generated")
    p.add_argument("--no-repeat-ngram-size", type=int, default=0,
                   help="ban any token completing an n-gram already generated; a hard "
                        "guarantee, but blunt on code, where short n-grams legitimately repeat")
    p.add_argument("--penalty-window", type=int, default=256,
                   help="how many recent generated tokens the penalties count over")
    p.add_argument("--system", default=None, help="System prompt.")
    p.add_argument("--no-typing-prefill", action="store_true",
                   help="Disable prefilling the message while it is being typed (interactive terminals only).")
    p.add_argument("prompt", nargs="*")

    p = sub.add_parser("doctor", help="Check hardware, storage, memory and checkpoint layout.")
    p.add_argument("--model", default=cfg.resolved_model_path)
    p.add_argument("--expert-budget-gib", type=_budget, default=0.0)

    p = sub.add_parser("bench", help="Run the routing-trace benchmark and print the analysis.")
    _runtime_args(p, cfg)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--decode-tokens", type=int, default=32)

    return parser


# ----------------------------------------------------------------------------
def _family(args) -> str:
    if getattr(args, "family", "auto") != "auto":
        return args.family
    try:
        import json

        config = json.loads((Path(args.model) / "config.json").read_text())
    except (OSError, ValueError, TypeError):
        return "deepseek"
    kind = config.get("model_type") or (config.get("text_config") or {}).get("model_type") or ""
    if kind.startswith("glm5_next"):
        return "glm"
    if kind.startswith("minimax_m3"):
        return "minimax"
    return "deepseek"


def _load_glm(args):
    from cachalot.glm.model import GlmModel

    print(f"cachalot {__version__}: loading GLM-5.3-Flash from {args.model}", file=sys.stderr, flush=True)
    budget = args.expert_budget_gib or 52.0
    model = GlmModel(args.model, expert_budget_gib=budget, load_workers=max(8, args.io_workers), verbose=args.verbose)
    model.max_seq_len = args.max_seq_len
    return model


def _load_minimax(args):
    from cachalot.minimax.model import MiniMaxModel

    print(f"cachalot {__version__}: loading MiniMax-M3 from {args.model}", file=sys.stderr, flush=True)
    budget = args.expert_budget_gib or 52.0
    model = MiniMaxModel(args.model, expert_budget_gib=budget, load_workers=max(8, args.io_workers), verbose=args.verbose)
    model.max_seq_len = args.max_seq_len
    return model


def _load_model(args):
    from cachalot.model.api import V41Model

    t0 = time.perf_counter()
    print(f"cachalot {__version__}: loading {args.model}", file=sys.stderr, flush=True)
    model = V41Model.from_pretrained(
        args.model,
        max_seq_len=args.max_seq_len,
        expert_cache_budget_bytes=int(args.expert_budget_gib * GiB),
        io_workers=args.io_workers,
        verbose=args.verbose,
    )
    rt = model.runtime
    print(f"ready in {time.perf_counter() - t0:.1f}s "
          f"(expert budget {rt.expert_cache_budget_bytes / GiB:.1f} GiB, wired {rt.mlx_wired_limit_bytes / GiB:.1f} GiB, "
          f"max_seq_len {args.max_seq_len})",
          file=sys.stderr, flush=True)
    return model


def _attach_snapshot_store(runtime, directory: str) -> None:
    """
    Load the system-prompt snapshots a previous server left in `directory`
    and write new ones there (HANDOFF section 15.4). Only the most recently
    used few are loaded now; the others load when a request needs them
    (section 15.12).
    """
    from cachalot.model import snapshot_store

    t0 = time.perf_counter()
    identity = snapshot_store.runtime_identity(
        runtime.model_path, runtime.expert_bank_path, runtime.max_seq_len, snapshot_store.NUMERICS_VERSION
    )
    store = snapshot_store.SnapshotStore(directory, identity)
    loaded = store.load_all()
    for snap in loaded:
        # boundary=True pins it; persist is not attached yet, so nothing is rewritten
        runtime.prefix_cache.add(snap, boundary=True)
    runtime.prefix_cache.persist = store.persist
    runtime.prefix_cache.on_find = store.on_find
    runtime.prefix_cache.fetch = store.fetch
    print(
        f"prefix snapshots: {len(loaded)} loaded from {directory} "
        f"({', '.join(str(len(s.tokens)) for s in loaded) or 'none'} tokens), "
        f"{len(store.tokens) - len(loaded)} more on disk, in {time.perf_counter() - t0:.2f}s",
        file=sys.stderr,
        flush=True,
    )


def cmd_serve(args) -> None:
    try:
        import uvicorn
    except ImportError:
        sys.exit("The server needs extra dependencies: pip install 'cachalot[server]'")

    from cachalot.server.app import ServerConfig, create_app
    from cachalot.server.engine import Engine

    if _family(args) in ("glm", "minimax"):
        from cachalot.glm.engine import GlmEngine

        family = _family(args)
        model = _load_minimax(args) if family == "minimax" else _load_glm(args)
        if args.snapshot_dir:
            print(model.attach_snapshot_store(args.snapshot_dir), file=sys.stderr, flush=True)
        default_id = "minimax-m3" if family == "minimax" else "glm-5.3-flash"
        engine = GlmEngine(model, model_id=args.model_id if args.model_id != "deepseek-v4.1-flash" else default_id)
    else:
        model = _load_model(args)
        if args.snapshot_dir:
            _attach_snapshot_store(model.runtime, args.snapshot_dir)
        engine = Engine(model, model_id=args.model_id)
    app = create_app(
        engine,
        ServerConfig(
            default_max_tokens=args.default_max_tokens,
            default_temperature=args.default_temperature,
            default_thinking=args.thinking,
            default_reasoning_effort=_effort(args.reasoning_effort),
            api_key=args.api_key,
        ),
    )
    print(f"OpenAI-compatible API on http://{args.host}:{args.port}/v1  (model id: {engine.model_id})",
          file=sys.stderr, flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
    finally:
        model.close()


def _effort(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def cmd_chat_glm(args, family: str = "glm") -> None:
    """Terminal chat for the streamed-expert families (GLM-5.3-Flash, MiniMax-M3)."""
    model = _load_minimax(args) if family == "minimax" else _load_glm(args)
    THINK_END = model.splitter_cls.THINK_END
    thinking = bool(args.thinking or args.reasoning_effort)
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    def turn(user_text: str) -> None:
        messages.append({"role": "user", "content": user_text})
        text = model.render_chat(messages, thinking=thinking, effort=args.reasoning_effort)
        ids = list(model.tokenizer.encode(text, add_special_tokens=False))
        splitter = model.splitter_cls(model.tokenizer, thinking)
        n, in_reasoning, t_start = 0, False, time.perf_counter()
        s0 = model.store.stats()
        for ev in model.stream(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature):
            if ev[0] == "prefill":
                fresh = len(ids) - ev[1]
                rate = f", {fresh / ev[2]:.1f} tok/s on {fresh} new" if fresh > 0 and ev[2] > 0 else ""
                print(f"[prefill {len(ids)} tokens, reused {ev[1]}, {ev[2]:.1f}s{rate}]", file=sys.stderr, flush=True)
            elif ev[0] == "token":
                n += 1
                if ev[1] in model.eos_ids:
                    continue
                d = splitter.push(ev[1])
                if d.reasoning:
                    if not in_reasoning:
                        print("\x1b[2m<think>", end="", flush=True)
                        in_reasoning = True
                    print(d.reasoning, end="", flush=True)
                if d.content:
                    if in_reasoning:
                        print("</think>\x1b[0m\n", end="", flush=True)
                        in_reasoning = False
                    print(d.content, end="", flush=True)
            elif ev[0] == "done":
                tail = splitter.flush()
                if in_reasoning:
                    print(tail.reasoning + "</think>\x1b[0m\n", end="", flush=True)
                print(tail.content, flush=True)
                s1 = model.store.stats()
                misses = s1.cache_misses - s0.cache_misses
                hits = s1.cache_hits - s0.cache_hits
                dt = ev[2]
                print(f"\n[{n} tokens, {n / dt if dt else 0:.2f} tok/s decode, {time.perf_counter() - t_start:.1f}s total, "
                      f"stop={ev[1]}, expert hit {hits / max(1, hits + misses):.0%}]", file=sys.stderr, flush=True)
        content = splitter.text.split(THINK_END, 1)[-1] if thinking else splitter.text
        messages.append({"role": "assistant", "content": content})

    try:
        if args.prompt:
            turn(" ".join(args.prompt))
            return
        while True:
            try:
                user = input("\n> ")
            except EOFError:
                break
            if user.strip() in ("/exit", "/quit"):
                break
            if user.strip():
                turn(user)
    finally:
        model.close()


def cmd_chat(args) -> None:
    if _family(args) in ("glm", "minimax"):
        return cmd_chat_glm(args, _family(args))
    from cachalot.model.generation import SamplingParams, load_official_encoding, stream_tokens
    from cachalot.server.engine import _TextSplitter

    model = _load_model(args)
    encoding = load_official_encoding(args.model)
    thinking_mode = "thinking" if (args.thinking or args.reasoning_effort) else "chat"
    params = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        penalty_window=args.penalty_window,
    )

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    def turn(user_text: str) -> None:
        messages.append({"role": "user", "content": user_text})
        prompt = encoding.encode_messages(messages, thinking_mode=thinking_mode, reasoning_effort=_effort(args.reasoning_effort))
        ids = list(model.tokenizer.encode(prompt))
        splitter = _TextSplitter(model.tokenizer, thinking_mode)
        n_tokens = 0
        in_reasoning = False
        t_start = time.perf_counter()
        for ev in stream_tokens(model.runtime, ids, params):
            if ev.kind == "prefill":
                # Rate the tokens actually computed, not the whole prompt: the
                # reused prefix costs nothing, so dividing by prompt_tokens
                # would report a speed the runtime never achieved.
                fresh = ev.prompt_tokens - ev.reused_prefix_tokens
                rate = (
                    f", {fresh / ev.prefill_seconds:.1f} tok/s on {fresh} new"
                    if fresh > 0 and ev.prefill_seconds > 0
                    else ""
                )
                print(f"[prefill {ev.prompt_tokens} tokens, reused {ev.reused_prefix_tokens}, "
                      f"{ev.prefill_seconds:.1f}s{rate}]",
                      file=sys.stderr, flush=True)
            elif ev.kind == "token":
                n_tokens += 1
                d = splitter.push(ev.token)
                if d.reasoning:
                    if not in_reasoning:
                        print("\x1b[2m<think>", end="", flush=True)
                        in_reasoning = True
                    print(d.reasoning, end="", flush=True)
                if d.content:
                    if in_reasoning:
                        print("</think>\x1b[0m\n", end="", flush=True)
                        in_reasoning = False
                    print(d.content, end="", flush=True)
            elif ev.kind == "done":
                tail = splitter.flush()
                if in_reasoning:
                    print("</think>\x1b[0m\n", end="", flush=True)
                print(tail.content, flush=True)
                dt = ev.decode_seconds
                print(f"\n[{n_tokens} tokens, {n_tokens / dt if dt else 0:.2f} tok/s decode, "
                      f"{time.perf_counter() - t_start:.1f}s total, stop={ev.finish_reason}]",
                      file=sys.stderr, flush=True)
        parsed = None
        try:
            parsed = encoding.parse_message_from_completion_text(
                splitter.text + model.tokenizer.eos_token, thinking_mode=thinking_mode)
        except Exception:
            pass
        messages.append(parsed or {"role": "assistant", "content": splitter.text})

    # Typing-time prefill (see cachalot.chat_input): while the user types on a
    # real terminal, the template head and the finished words of the message
    # are prefilled in the background so Enter only pays for the last word.
    from cachalot.chat_input import RawLineReader, SpeculativePrefiller, split_prompt_template

    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not args.no_typing_prefill
    prefiller = SpeculativePrefiller(model.runtime, model.tokenizer, verbose=args.verbose) if interactive else None
    reader = RawLineReader(on_idle=prefiller.update if prefiller else None) if interactive else None

    def read_line() -> str:
        if reader is None:
            return input("\n>>> ").strip()
        head, _tail = split_prompt_template(
            encoding, messages, thinking_mode=thinking_mode, reasoning_effort=_effort(args.reasoning_effort))
        prefiller.set_context(head)
        text = reader.readline("\n>>> ").strip()
        prefiller.wait_idle()
        return text

    try:
        if args.prompt:
            turn(" ".join(args.prompt))
            return
        print("Cachalot chat. Commands: /clear, /stats, /exit", file=sys.stderr)
        while True:
            try:
                text = read_line()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text == "/exit":
                break
            if text == "/clear":
                messages[:] = [m for m in messages if m["role"] == "system"]
                print("conversation cleared", file=sys.stderr)
                continue
            if text == "/stats":
                stats = model.stats()
                if prefiller is not None:
                    stats["typing_prefill_runs"] = prefiller.runs
                    stats["typing_prefill_tokens"] = prefiller.prefilled_tokens
                    stats["typing_prefill_seconds"] = round(prefiller.seconds, 2)
                print(json.dumps(stats, indent=2))
                continue
            turn(text)
    finally:
        if prefiller is not None:
            prefiller.close()
        model.close()


def cmd_doctor(args) -> None:
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        mark = "OK " if passed else "FAIL"
        print(f"[{mark}] {label}{': ' + detail if detail else ''}")

    print(f"cachalot {__version__} doctor\n")
    check("platform", platform.machine() == "arm64" and sys.platform == "darwin",
          f"{platform.platform()} ({platform.machine()})")
    check("python", sys.version_info >= (3, 12), platform.python_version())
    try:
        import mlx.core as mx

        info = mx.device_info()
        mem_gib = info["memory_size"] / GiB
        check("mlx", True, f"{mx.__version__} on {info['device_name']}")
        check("unified memory >= 48 GiB", mem_gib >= 48, f"{mem_gib:.0f} GiB")
        from dataclasses import replace

        from cachalot.config import resolve_expert_budget, resolve_wired_limit

        cfg2 = replace(load_config(), expert_cache_budget_bytes=int(args.expert_budget_gib * GiB))
        budget = resolve_expert_budget(cfg2)
        wired = resolve_wired_limit(cfg2, budget)
        share = budget / (15_360 * 18_800_640)
        check("expert budget", budget >= 8 * GiB,
              f"{budget / GiB:.1f} GiB ({share:.0%} of all experts) "
              f"{'auto' if args.expert_budget_gib == 0 else 'requested'}; wired limit {wired / GiB:.1f} GiB "
              f"of {info['max_recommended_working_set_size'] / GiB:.1f} GiB recommended")
        if budget >= 15_360 * 18_800_640:
            print("       all routed experts fit in memory: SSD is only touched at load time")
    except Exception as exc:  # pragma: no cover
        check("mlx", False, str(exc))

    model = Path(args.model)
    check("checkpoint directory", model.is_dir(), str(model))
    if model.is_dir():
        shards = sorted(model.glob("model-*.safetensors"))
        size = sum(p.stat().st_size for p in shards)
        check("safetensors shards", len(shards) == 48, f"{len(shards)} shards, {size / 1e9:.0f} GB")
        check("config.json", (model / "config.json").exists())
        check("encoding/encoding.py (official chat protocol)", (model / "encoding" / "encoding.py").exists())
        check("tokenizer", (model / "tokenizer.json").exists() or (model / "tokenizer_config.json").exists())
        usage = shutil.disk_usage(model)
        print(f"       volume free: {usage.free / GiB:.0f} GiB")
        if shards:
            speed = _read_speed(shards[len(shards) // 2])
            grade = "internal/NVMe class" if speed > 2500 else "Thunderbolt class" if speed > 1500 else "USB 3.2 class"
            check("storage read speed", speed > 400, f"{speed:.0f} MB/s ({grade}); decode misses cost ~{18.8 * 1000 / max(speed, 1):.0f} ms each")
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout.strip()
        print(f"       swap: {out}")
    except Exception:
        pass
    print("\nall checks passed" if ok else "\nsome checks failed", file=sys.stderr)
    sys.exit(0 if ok else 1)


def _read_speed(path: Path, total_bytes: int = 1 << 30) -> float:
    """Sequential read of ~1 GiB from the middle of a shard, MB/s."""
    import fcntl

    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        except OSError:
            pass
        size = os.fstat(fd).st_size
        offset = max(0, (size // 2) & ~0xFFFF)
        chunk = 16 << 20
        read = 0
        t0 = time.perf_counter()
        while read < total_bytes and offset + read < size:
            data = os.pread(fd, chunk, offset + read)
            if not data:
                break
            read += len(data)
        dt = time.perf_counter() - t0
        return read / dt / 1e6 if dt else 0.0
    finally:
        os.close(fd)


def cmd_bench(args) -> None:
    bench_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    script = bench_dir / "trace_routing.py"
    if not script.exists():
        sys.exit("benchmarks/ not found; run from a source checkout")
    cmd = [sys.executable, str(script), "--model", args.model, "--prompt-tokens", str(args.prompt_tokens),
           "--decode-tokens", str(args.decode_tokens), "--expert-budget-gib", str(args.expert_budget_gib),
           "--io-workers", str(args.io_workers)]
    subprocess.run(cmd, check=True)
    trace = bench_dir / "results" / "trace_routing.trace.npz"
    subprocess.run([sys.executable, str(bench_dir / "analyze_trace.py"), str(trace),
                    "--out", str(bench_dir / "results" / "trace_routing.md")], check=True)


def default_fast_synch() -> str:
    """Default MLX to shared-memory Metal fences unless the environment says otherwise.

    MLX reads MLX_METAL_FAST_SYNCH once, on its first Metal fence, so this runs before any model work
    (serve.sh and chat.sh also export it). Bit-identical output; faster decode in a display-on slow
    window (HANDOFF section 15.9). MLX_METAL_FAST_SYNCH=0 disables.
    """
    return os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command in ("serve", "chat"):
        print(f"MLX_METAL_FAST_SYNCH={default_fast_synch()}", file=sys.stderr, flush=True)
        from cachalot.darwin_role import apply_darwin_role

        role = apply_darwin_role()
        if role is not None:
            print(f"darwin role {role} (CACHALOT_DARWIN_ROLE=0 disables)", file=sys.stderr, flush=True)
        from cachalot.model.vision_ablation import announce

        announce()
    {"serve": cmd_serve, "chat": cmd_chat, "doctor": cmd_doctor, "bench": cmd_bench}[args.command](args)


if __name__ == "__main__":
    main()
