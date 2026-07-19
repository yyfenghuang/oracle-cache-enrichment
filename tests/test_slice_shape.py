"""Shape and content correctness for slice_cache(), across both models'
actual head configurations.

This repo currently runs Qwen3-0.6B alone (see MODEL_ID in
run_experiment.py). Both configs are tested anyway, because cache_enrichment.py
is written to be reused unchanged in Misi 1's two-model transfer
(Qwen2.5-0.5B Sharer, Qwen3-0.6B Receiver), and a shape assumption that only
happens to hold for one head configuration should be caught now rather than
mid-Misi-1. Values below are taken directly from each model's published
config.json, not guessed:

  Qwen2.5-0.5B: num_hidden_layers=24, num_attention_heads=14,
  num_key_value_heads=2, hidden_size=896. head_dim is not an explicit config
  field for this architecture; transformers derives it as
  hidden_size // num_attention_heads = 64.

  Qwen3-0.6B: num_hidden_layers=28, num_attention_heads=16,
  num_key_value_heads=8, hidden_size=1024, head_dim=128 (an explicit config
  field for this architecture, and notably NOT equal to
  hidden_size // num_attention_heads = 64 - Qwen3 decouples head_dim from
  that ratio). Confirm both figures against `model.config` directly before
  trusting them for anything beyond this test; published config.json can
  drift from what a given transformers version actually loads.

No model or tokenizer is loaded. Caches are built directly from random
tensors, which is all slice_cache() ever touches.
"""

import torch
from transformers.cache_utils import DynamicCache

from cache_enrichment import slice_cache

QWEN2_5_0_5B = dict(num_hidden_layers=24, num_kv_heads=2, head_dim=64)
QWEN3_0_6B = dict(num_hidden_layers=28, num_kv_heads=8, head_dim=128)
CONFIGS = {"qwen2.5-0.5b": QWEN2_5_0_5B, "qwen3-0.6b": QWEN3_0_6B}

BATCH = 1


def _make_cache(cfg, total_len, dtype=torch.float32):
    """Build a synthetic cache where position i's key/value at every layer
    and every kv head is filled with the constant `i`. This makes a wrong
    slice window (off by one, reversed, wrong length) show up as a value
    mismatch, not just a shape mismatch.
    """
    layers = []
    for _ in range(cfg["num_hidden_layers"]):
        k = torch.zeros(BATCH, cfg["num_kv_heads"], total_len, cfg["head_dim"], dtype=dtype)
        v = torch.zeros(BATCH, cfg["num_kv_heads"], total_len, cfg["head_dim"], dtype=dtype)
        for i in range(total_len):
            k[..., i, :] = i
            v[..., i, :] = i
        layers.append((k, v))
    return DynamicCache(ddp_cache_data=layers)


results = []


def check(label, condition, note=""):
    status = "PASS" if condition else "FAIL"
    results.append(status == "PASS")
    print(f"[{status}] {label}" + (f"  ({note})" if note else ""))


# --- shape and content, both real configs, several (len_E, len_X) pairs ---

for config_name, cfg in CONFIGS.items():
    for len_E, len_X in [(0, 5), (1, 1), (528, 100), (17, 390)]:
        total_len = len_E + len_X
        cache = _make_cache(cfg, total_len)
        sliced = slice_cache(cache, len_E, len_X)

        label_base = f"{config_name} len_E={len_E} len_X={len_X}"

        check(
            f"{label_base}: layer count preserved",
            len(sliced.layers) == cfg["num_hidden_layers"],
        )

        all_layers_ok = True
        for layer_idx in range(cfg["num_hidden_layers"]):
            k, v = sliced.layers[layer_idx].keys, sliced.layers[layer_idx].values
            shape_ok = (
                k.shape == (BATCH, cfg["num_kv_heads"], len_X, cfg["head_dim"])
                and v.shape == (BATCH, cfg["num_kv_heads"], len_X, cfg["head_dim"])
            )
            expected = torch.arange(len_E, len_E + len_X, dtype=k.dtype)
            content_ok = bool(
                torch.all(k[..., :] == expected.view(1, 1, len_X, 1))
                and torch.all(v[..., :] == expected.view(1, 1, len_X, 1))
            )
            all_layers_ok = all_layers_ok and shape_ok and content_ok
        check(
            f"{label_base}: shape and content correct at every layer",
            all_layers_ok,
            "content check catches an off-by-one or reversed slice that a shape-only check would miss",
        )

# --- clone, not view: mutating the source cache must not affect the slice ---

cfg = QWEN3_0_6B
cache = _make_cache(cfg, total_len=50)
sliced = slice_cache(cache, len_E=10, len_X=20)
before = sliced.layers[0].keys.clone()
cache.layers[0].keys[..., 15, :] = -999.0  # mutate the source in the slice's window
check(
    "clone not view: mutating source cache after slicing leaves the slice unchanged",
    torch.equal(sliced.layers[0].keys, before),
    "a view would keep the parent tensor alive and this mutation would leak through",
)

# --- error handling ---

cfg = QWEN3_0_6B
cache = _make_cache(cfg, total_len=10)

try:
    slice_cache(cache, len_E=-1, len_X=5)
    check("len_E negative raises ValueError", False)
except ValueError:
    check("len_E negative raises ValueError", True)

try:
    slice_cache(cache, len_E=0, len_X=0)
    check("len_X zero raises ValueError", False)
except ValueError:
    check("len_X zero raises ValueError", True)

try:
    slice_cache(cache, len_E=8, len_X=5)  # needs 13, cache only has 10
    check("slice exceeding cache length raises ValueError", False)
except ValueError:
    check("slice exceeding cache length raises ValueError", True)

empty_cache = DynamicCache(ddp_cache_data=[])
try:
    slice_cache(empty_cache, len_E=0, len_X=5)
    check("empty cache (no layers) raises ValueError", False)
except ValueError:
    check("empty cache (no layers) raises ValueError", True)

print()
print(f"{sum(results)}/{len(results)} passed")
