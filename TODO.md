# TODO

Ordered within each block. A checked item does not mean polished, it means the gate it represents has been passed. Cross-block dependencies are noted where they exist.

## Scripting

- [ ] `cache_enrichment.py`: `slice_cache(cache, len_E, len_X)`, pure function, no I/O
- [ ] `cache_enrichment.py`: `correct_position_ids(len_E, len_X)`, returns absolute positions `len_E .. len_E+len_X-1`
- [ ] `extractor.py`: parser for Direct (tolerates leading prose or punctuation before the answer letter)
- [ ] `extractor.py`: parser for Few-shot (stops before the model invents a new question)
- [ ] `extractor.py`: parser for Oracle (tolerates degeneration into a short loop after the answer)
- [ ] `run_experiment.py`: loads both conditions, calls `cache_enrichment.py` and `extractor.py`, writes one row per sample per condition to `results/run_<date>.json`
- [ ] `analysis.py`: paired accuracy difference, bootstrap CI, McNemar statistic
- [ ] `analysis.py`: classifies the run into outcome A, B, or C per the criteria in `README.md`, prints the classification, exits 0 regardless of which outcome

## Testing

- [ ] Hand-label a small probe set per condition (Direct, Few-shot, Oracle) by reading raw model output directly, before writing any parser. Save as `probe_labels.json`
- [ ] `test_slice_shape.py`: sliced cache has the expected shape at every layer, for both models' head configurations
- [ ] `test_position_correction.py`: regression test, position ids after slicing equal `len_E + i`, not `i`
- [ ] `test_extractor_probe.py`: each parser scored against `probe_labels.json`, fails below an explicit precision threshold
- [ ] Full run (`run_experiment.py --n 100`) only after the three test files above pass
- [ ] `analysis.py` run on the resulting `results/run_<date>.json`, outcome recorded in `README.md` or a linked note

## Visualization

Each one drafted in `sandbox/` first, promoted to `assets/` only once finished. Tag shows static or animated.

- [ ] 1. [static] Three-condition cache-length diagram (Direct, Few-shot, Oracle)
- [ ] 2. [animated] Contamination of `h_X` accumulating across layers
- [ ] 3. [static] Same token X, different `K_X` depending on what preceded it
- [ ] 4. [animated] Cosine distance between `K_X(X)` and `K_X(E⊕X)`, swept across layers — depends on `cache_enrichment.py`
- [ ] 5. [static] Tensor slicing diagram, cache array with the `|E|` and `|E|+|X|` cut lines
- [ ] 6. [animated] RoPE position bug, rotation with correct vs reset absolute position
- [ ] 7. [static] Attention score heatmap, correct position vs corrupted position, side by side
- [ ] 8. [animated] Evaluation-mode collapse, Oracle and Few-shot pipelines converging on the same last-position logits
- [ ] 9. [static] McNemar contingency table — template only until a real run exists, populated from `analysis.py` output
- [ ] 10. [static] Bootstrap distribution histogram — template only until a real run exists, populated from `analysis.py` output

## Notebook synthesis

- [ ] Stage 1, Intuition and Problem: the H1/H2 decomposition, visual 1
- [ ] Stage 2, Derivation without hand-waving: mechanistic argument for why a trace of E can persist in `K_X`, explicit note that this argument does not establish the trace is large enough to matter, visuals 2 to 4
- [ ] Stage 3, Math to tensor bridge: symbol-to-code table, visual 5
- [ ] Stage 4, Bug autopsy: RoPE position bug and evaluation-mode collapse, written as derivations where possible rather than narration alone, visuals 6 to 8
- [ ] Stage 5, Measurement: outcome A, B, or C, visuals 9 and 10, the only stage allowed to contain an accuracy number
- [ ] Pass over the finished notebook: confirm no accuracy number appears before Stage 5
- [ ] Where the notebook shows code, use `inspect.getsource()` against the canonical module in `cache_enrichment.py` or `extractor.py`, with a drift-guard assertion, rather than retyping the function body
