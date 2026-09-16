from cachalot.chat_input import safe_partial, split_prompt_template


class FakeEncoding:
    def encode_messages(self, messages, *, thinking_mode, reasoning_effort):
        body = "".join(f"<{m['role']}>{m['content']}<eos>" for m in messages)
        return body + "<assistant>"


def test_split_prompt_template_recovers_head_and_tail():
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    head, tail = split_prompt_template(FakeEncoding(), history, thinking_mode="chat", reasoning_effort=None)
    full = FakeEncoding().encode_messages(history + [{"role": "user", "content": "new text"}],
                                          thinking_mode="chat", reasoning_effort=None)
    assert head + "new text" + tail == full
    assert head.endswith("<user>")
    assert tail == "<eos><assistant>"


def test_safe_partial_cuts_only_at_single_inner_spaces():
    assert safe_partial("write a story of") == "write a story"
    assert safe_partial("write a story of a") == "write a story of"
    assert safe_partial("write") == ""
    assert safe_partial("write ") == ""            # trailing space: last word may still grow
    assert safe_partial("write  a") == ""          # double space merges into one token
    assert safe_partial("a\nb c") == "a\nb"        # newline is never a cut point, single space is
    assert safe_partial("hello world, again") == "hello world,"


def test_safe_partial_skips_commands():
    assert safe_partial("/stats") is None
    assert safe_partial("/clear now") is None
    assert safe_partial("") == ""
