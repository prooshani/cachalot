from __future__ import annotations

import argparse

from cachalot.model.api import (
    DEFAULT_MODEL_PATH,
    V41Model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v41-chat",
        description=(
            "Run DeepSeek-V4.1 locally through the "
            "cachalot execution engine."
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the DeepSeek-V4.1 checkpoint.",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--thinking-mode",
        choices=("chat", "thinking"),
        default="chat",
    )

    parser.add_argument(
        "--reasoning-effort",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable runtime and generation diagnostics.",
    )

    parser.add_argument(
        "prompt",
        nargs="*",
        help=(
            "Prompt text. If omitted, an interactive "
            "chat session is started."
        ),
    )

    return parser


def run_single_prompt(
    model: V41Model,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    thinking_mode: str,
    reasoning_effort: int | None,
    seed: int | None,
) -> None:
    response = model.chat(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        seed=seed,
        reset=True,
    )

    print()
    print(response.content)


def run_interactive(
    model: V41Model,
    *,
    max_new_tokens: int,
    temperature: float,
    thinking_mode: str,
    reasoning_effort: int | None,
    seed: int | None,
) -> None:
    messages: list[dict[str, str]] = []

    print()
    print("DeepSeek-V4.1 local chat")
    print("Commands: /clear, /exit")
    print()

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue

        if prompt == "/exit":
            break

        if prompt == "/clear":
            messages.clear()
            print("Conversation cleared.")
            continue

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        response = model.chat(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            seed=seed,
            reset=True,
        )

        print()
        print(response.content)
        print()

        if response.message is not None:
            messages.append(
                response.message
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )


def main() -> None:
    args = build_parser().parse_args()

    with V41Model.from_pretrained(
        args.model,
        max_seq_len=args.max_seq_len,
        verbose=args.verbose,
    ) as model:
        if args.prompt:
            run_single_prompt(
                model,
                " ".join(args.prompt),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                thinking_mode=args.thinking_mode,
                reasoning_effort=args.reasoning_effort,
                seed=args.seed,
            )
        else:
            run_interactive(
                model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                thinking_mode=args.thinking_mode,
                reasoning_effort=args.reasoning_effort,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
