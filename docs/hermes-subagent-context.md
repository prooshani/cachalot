# Hermes subagents and the prefix cache: where the per-task context goes

Written 2026-09-24 (Cachalot 0.16.0, HANDOFF section 15.12). Hermes checkout read: `~/.hermes/hermes-agent`
at `074349fb27` (2026-09-24). Nothing in Hamed's Hermes install was changed.

## The cost today

Each `delegate_task` subagent sends a first request of ~19.7-19.9k tokens. In Hamed's session of 2026-09-24
(HANDOFF section 15.10) six were dispatched at once. Their prompts share the first 5,663 tokens (Hermes's base
system prompt) and then diverge at the subagent's `CONTEXT:` section; after it come ~13.5k tokens of tool
schemas that are byte-identical across all six (same 22 tools, same JSON), and a short user turn (the goal).
A prefix cache cannot reach identical tokens behind a divergence, so every subagent's first turn prefilled
~15.6k tokens (`reused=4096`), about 190 s each on this machine.

## Why it looks like that

`tools/delegate_tool_progress.py::_build_child_system_prompt` builds the child's ephemeral system prompt:

```
You are a focused subagent working on a specific delegated task.
CONTEXT: <per task>
WORKSPACE PATH: <per parent>
<project context files, if any>
<completion instructions, constant>
[orchestrator block, constant per depth]
```

`agent/chat_completion_helpers.py` appends it to the base system prompt (`effective_system + "\n\n" +
ephemeral_system_prompt`), and the server's chat template renders the request's `tools` after the system
message. So the only per-task text sits in the middle of the system message, in front of the tool schemas.
The goal itself is already the child's first user turn (`tools/delegate_tool_child_run.py`,
`_ChildRun.await_child`; the source comment says it was moved there to avoid sending the task in both roles).

**There is no configuration option for this.** `delegation:` in `config.yaml` covers credentials, model,
reasoning effort, concurrency, timeouts, approval, worktree isolation and orchestration, but not where the
context is placed.

## What moving it would buy

`benchmarks/prefix_pin_replay.py` on Hamed's session (requests 130-204 of the dump), once as recorded and
once with each subagent's `CONTEXT:` ... text moved from the system message to the front of its first user
message (a scratch transform of the dump; the prefix cache and the prompt rendering are the real ones):

| subagent first turn | as recorded: reused / prefilled | context in the user turn: reused / prefilled |
|---|---|---|
| 1st | 0 / 19,887 | 0 / 19,887 |
| 2nd-6th | 4,096 / 15,615-15,799 | 19,362 / 349-533 |

**About 76k fewer prefilled tokens per six-subagent batch, roughly 15 minutes at 12.3 ms/token.** All
subagents would share one 19,362-token system block, so the disk snapshot store would also keep one file for
them instead of one per task, and the first subagent of the next batch (or after a restart) would reuse it
too.

## The change, for Hermes (not applied)

Keep the ephemeral system prompt constant and put the per-task parts in the user turn:

1. `_build_child_system_prompt`: build only the constant parts (intro, completion instructions, orchestrator
   block) and return the per-task parts (CONTEXT, WORKSPACE PATH, context files) separately.
2. `_ChildRun.await_child`: prepend those parts to the goal when it builds `user_message` (the text part of a
   multimodal goal, as `_build_child_goal_message` already does for the worktree note).

Caveats worth raising with the Hermes maintainers: project context files are introduced as "binding"; in the
user role they may carry less weight with some models. The workspace path could stay in the system prompt when
all children of one parent share it (it is per parent, not per task), which keeps the block shared. The same
reordering helps every provider with prompt caching (Anthropic, OpenAI, DeepSeek's own API), not only a local
server, so it is a reasonable upstream proposal rather than a Cachalot-only patch.

Not possible on the server side: rendering the tools before the system text would change the chat template the
model was trained on, and reusing the tool-schema span across a divergence would need position-independent KV,
which causal attention does not allow (v44 Job 3: do not start it).
