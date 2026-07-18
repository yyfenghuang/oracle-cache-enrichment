"""Proves, with a stub model, that decode_loop's first generated token is
fixed entirely by `first_logits` and cannot be changed by anything the model
does afterward.

This is the reason the extract_oracle "same-line loop before the answer"
scenario raised in tests/test_extractor_synthetic.py cannot actually occur in
this harness: decode_loop appends argmax(first_logits) as generated token 0
BEFORE the loop body ever calls the model. Few-shot and Oracle are handed the
exact same `first_logits` tensor in run_sample(), so their first generated
token - the MCQ answer letter - is identical by construction, regardless of
what the sliced cache contains or how the model behaves on later steps.

No Qwen weights are needed for this: the claim being tested is about the
control flow of decode_loop, not about what Qwen specifically outputs, so a
stub model that deliberately tries to produce a *different* first token is
enough to show it cannot succeed.
"""

import torch

from run_experiment import decode_loop

RIGHT_TOKEN = 7   # what first_logits actually favors
WRONG_TOKEN = 3   # what every later step's logits favor, to try to interfere
VOCAB = 10


class _FakeCache:
    def __init__(self, seq_length):
        self._seq_length = seq_length

    def get_seq_length(self):
        return self._seq_length


class _FakeOut:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class _AdversarialModel:
    """Every forward pass after the first token strongly favors WRONG_TOKEN,
    as if the model were trying to overwrite the answer. decode_loop must
    never consult this for the first generated token."""

    def __call__(self, input_ids, position_ids, cache_position, past_key_values, use_cache):
        logits = torch.full((1, 1, VOCAB), -100.0)
        logits[0, 0, WRONG_TOKEN] = 100.0
        return _FakeOut(logits, _FakeCache(past_key_values.get_seq_length() + 1))


class _FakeTokenizer:
    eos_token_id = -1  # sentinel, never produced by this stub

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(i) for i in ids)


def test_first_token_is_fixed_by_first_logits_alone():
    first_logits = torch.full((1, 1, VOCAB), -100.0)
    first_logits[0, 0, RIGHT_TOKEN] = 100.0  # this is what the "answer" is

    model = _AdversarialModel()
    tokenizer = _FakeTokenizer()
    cache = _FakeCache(seq_length=50)

    text, n_generated = decode_loop(
        model, tokenizer, cache, first_logits, start_position=50, max_new_tokens=3
    )

    generated_ids = [int(t) for t in text.split(",")]
    assert generated_ids[0] == RIGHT_TOKEN, (
        f"token 0 must equal argmax(first_logits) = {RIGHT_TOKEN}, "
        f"got {generated_ids[0]}. If this ever fails, the structural "
        f"guarantee that Oracle and Few-shot share the same first answer "
        f"token no longer holds, and extract_oracle needs a real cut for "
        f"same-line loops preceding the answer."
    )
    # Everything after token 0 is free to be whatever the model prefers -
    # that is exactly where Oracle's degeneration is expected to live.
    assert generated_ids[1] == WRONG_TOKEN
    assert generated_ids[2] == WRONG_TOKEN
    print(f"[PASS] first_token_is_fixed_by_first_logits_alone: generated={generated_ids}")


if __name__ == "__main__":
    test_first_token_is_fixed_by_first_logits_alone()
