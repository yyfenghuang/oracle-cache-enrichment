"""Regression tests for correct_position_ids() and next_position_id().

The bug these guard against is silent by construction: resetting sliced
positions to 0 .. len_X - 1 instead of len_E .. len_E + len_X - 1 does not
crash, does not produce NaN, and does not change tensor shape. RoPE rotation
is orthogonal, so norms are preserved either way. Only the attention scores
are wrong, and only a downstream accuracy number would ever reveal it, long
after the mistake was made. A shape-only test would pass on the buggy
version just as readily as the correct one; every assertion here checks the
actual position values, not just their shape or dtype.
"""

import torch

from cache_enrichment import correct_position_ids, next_position_id

results = []


def check(label, condition, note=""):
    status = "PASS" if condition else "FAIL"
    results.append(status == "PASS")
    print(f"[{status}] {label}" + (f"  ({note})" if note else ""))


# --- correct_position_ids: values, not just shape ---

for len_E, len_X in [(0, 5), (1, 1), (528, 100), (17, 390), (2440, 301)]:
    ids = correct_position_ids(len_E, len_X)
    expected = list(range(len_E, len_E + len_X))
    got = ids.squeeze(0).tolist()

    check(
        f"len_E={len_E} len_X={len_X}: shape is [1, len_X]",
        tuple(ids.shape) == (1, len_X),
    )
    check(
        f"len_E={len_E} len_X={len_X}: positions start at len_E, not 0",
        got == expected,
        "the exact regression this test exists for: resetting to 0..len_X-1 "
        "instead of len_E..len_E+len_X-1 would still pass a shape-only check",
    )
    check(
        f"len_E={len_E} len_X={len_X}: dtype is long",
        ids.dtype == torch.long,
        "RoPE indexing expects integer positions; a float tensor would not "
        "raise here but would corrupt the rotation silently, the same way "
        "the wrong offset does",
    )

# --- explicit non-zero len_E: the exact shape of the historical bug ---

ids = correct_position_ids(len_E=528, len_X=100)
check(
    "positions do not start at 0 when len_E > 0",
    ids[0, 0].item() == 528 and ids[0, 0].item() != 0,
)
check(
    "last position is len_E + len_X - 1",
    ids[0, -1].item() == 528 + 100 - 1,
)

# --- device propagation ---

ids_cpu = correct_position_ids(10, 5, device="cpu")
check("device='cpu' is honored", ids_cpu.device.type == "cpu")

# --- next_position_id: the decode-step counterpart of the same bug ---

for len_E, len_X in [(0, 5), (528, 100), (2440, 301)]:
    pos = next_position_id(len_E, len_X)
    check(
        f"next_position_id(len_E={len_E}, len_X={len_X}) == len_E + len_X",
        pos == len_E + len_X,
    )
    check(
        f"next_position_id(len_E={len_E}, len_X={len_X}) != len_X alone",
        pos != len_X or len_E == 0,
        "guards the specific historical bug shape: using len_X instead of "
        "len_E + len_X as the first decode position",
    )

# --- the invariant tying slice and position together, matching
#     run_experiment_guide.md's own framing: oracle_cache == len_X and
#     oracle_pos == len_E + len_X, so their difference is exactly len_E ---

for len_E, len_X in [(411, 78), (1906, 390), (304, 61)]:
    oracle_cache = len_X  # what slice_cache() leaves in the cache
    oracle_pos = next_position_id(len_E, len_X)
    check(
        f"len_E={len_E} len_X={len_X}: oracle_pos - oracle_cache == len_E",
        oracle_pos - oracle_cache == len_E,
    )

# --- error handling, both functions ---

for fn, name in [(correct_position_ids, "correct_position_ids"), (next_position_id, "next_position_id")]:
    try:
        fn(-1, 5)
        check(f"{name}: len_E negative raises ValueError", False)
    except ValueError:
        check(f"{name}: len_E negative raises ValueError", True)

    try:
        fn(0, 0)
        check(f"{name}: len_X zero raises ValueError", False)
    except ValueError:
        check(f"{name}: len_X zero raises ValueError", True)

    try:
        fn(0, -3)
        check(f"{name}: len_X negative raises ValueError", False)
    except ValueError:
        check(f"{name}: len_X negative raises ValueError", True)

print()
print(f"{sum(results)}/{len(results)} passed")
