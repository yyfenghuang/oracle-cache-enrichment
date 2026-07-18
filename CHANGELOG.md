# Changelog

Notable events from Misi 0 execution are logged here in the order they
happened, including the false starts. A finding that later turned out not to
be a bug is kept in this log with the correction attached, not deleted,
because the reasoning that got there is part of the record. No full run
(`--n 100` or larger) has happened yet; everything below comes from two
probes (`--probe 5`, `--probe 20`) and code review.

## 2026-07-19

### Fixed

- **`extractor.py` import failure.** The first probe attempt (`--probe 5`)
  crashed with `ImportError: cannot import name 'EXTRACTORS' from
  'extractor'`. The local edit to `extractor.py` had not been saved to disk
  before the run. Resolved once the file was saved.

- **`extract_direct` silently mis-extracts when the model restates the
  answer choices.** Confirmed with a synthetic case, not the probe:
  `"A. Paris\nB. London\nC. Berlin\nD. Madrid\nThe correct answer is B."`
  returned `A` instead of `B`, because `extract_direct` had no cut logic and
  took the first standalone letter regardless of where it came from. This
  matches a risk already named in `run_experiment_guide.md`'s anomaly table,
  but none of the 20 hand-read probe samples happened to trigger it, so the
  probe alone would never have caught it. Fixed with `_skip_choice_list`,
  which skips a block of two or more consecutive choice-shaped lines
  (`"A. ..."`, `"B) ..."`, and so on) at the start of the output before
  searching for the answer letter. The two-line threshold was chosen
  specifically so a genuine one-line answer like `"B. This is because..."`
  is never skipped. Verified against all 20 real Direct probe samples: no
  regressions.

- **`probe.log` tracked in git despite being listed in `.gitignore`.**
  `.gitignore` only stops new files from being tracked; `probe.log` had
  already been committed once before the rule existed. Untracked with
  `git rm --cached probe.log`, kept on disk locally. History before this
  commit still contains it; left as is, since the content is only
  dataset-loading logs and carries no credentials.

### Discovered, then corrected

- **Oracle's extracted answer letter is identical to Few-shot's, in all 20
  probed samples.** First read as evidence that slicing the cache has no
  effect on the measured outcome at all, which would have been a serious
  threat to the mission's validity. Traced to `run_sample()` /
  `decode_loop()`: both conditions decode from the same `first_logits`,
  computed once during the shared, unsliced E⊕X prefill, before Oracle's
  cache is even sliced. The first generated token, which is the MCQ answer
  letter, is therefore identical for both conditions by construction,
  regardless of what the slice does.

  On review this is not a flaw in the harness. It is the claim under test.
  Misi 0's actual question is whether Oracle, having discarded E's tokens
  from the cache, can perform like Few-shot (E⊕X kept in full) rather than
  like Direct (X alone). Oracle inheriting Few-shot's answer for free, then
  diverging only from the second generated token onward, is consistent with
  that claim rather than a contradiction of it. The valid accuracy
  comparison this harness supports is Direct vs. {Few-shot, Oracle} on the
  answer letter. Oracle's distinct value shows up in efficiency, which was
  already validated separately (about 4.4x faster decode, about 7x fewer
  cache tokens than Few-shot).

- **`extract_oracle`'s same-line loop blind spot.** `_cut_at_loop` only
  detects a repeating *line*, so a same-line degenerate loop (`"osten..."`,
  `"polers polers polers..."`, both seen in the real probe) is never cut.
  First treated as an unverified risk: if degenerate content ever appeared
  *before* the answer letter, `extract_oracle` would silently return the
  wrong one. Formally tested with a stub model
  (`tests/test_decode_loop_first_token_invariant.py`) that aggressively
  tries to force a different first token on every step after the first: the
  forced token never appears in position 0. `decode_loop` appends
  `argmax(first_logits)` as the first generated token before its loop body
  ever calls the model, so no downstream degeneration, same-line or
  otherwise, can precede the answer for Oracle or Few-shot. No change was
  made to `extractor.py` for this one; the adversarial synthetic case stays
  in the test file as a tripwire in case `decode_loop`'s control flow ever
  changes.

### Verified

- **Slice arithmetic and position correction**, across all 5 samples in the
  first probe: `oracle_cache == len_X` and `oracle_pos == len_E + len_X`
  held exactly in every case, with `len_E` and `len_X` varying per sample
  rather than sitting at a constant.

- **Extraction correctness against hand-read raw output**, across 60 data
  points (20 samples times 3 conditions, second probe): every `extracted`
  value matched what a human reading the `raw:` line would judge the
  model's answer to be. This included four distinct degeneration flavors:
  single-token loop, full-clause repetition, question-invention (seen in
  both Few-shot and, newly, Direct), and a repeated phrase lifted from the
  question itself.

### Added

- `probe_labels.json`: hand-labeled ground truth for 20 probe samples
  across all three conditions, with notes on which samples exercise
  adversarial shapes (Direct hallucinating a new question, Few-shot
  self-contradicting before hallucinating, and cases where conditions
  genuinely disagree on the answer rather than failing to parse).
- `tests/test_extractor_synthetic.py`: 11 hand-constructed cases, controls
  and the two risks above, that do not depend on what the model happens to
  say on a given run.
- `tests/test_decode_loop_first_token_invariant.py`: formal proof, via a
  stub model, of the first-token invariant described above.

### Not yet done

- `test_extractor_probe.py`, `test_slice_shape.py`,
  `test_position_correction.py` (TODO.md Testing block).
- `--n 100` run, gated on the three test files above passing.
- Full `--subject all` run.
