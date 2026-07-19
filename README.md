# Cache Enrichment First Principle

This minirepo proves one mechanism from scratch: whether a cache enriched by context that has since been discarded still carries information about that context. The Cache-to-Cache paper (C2C, ICLR 2026, arXiv:2510.03215) is where this question was first raised explicitly.  This document exists so that boundary does not quietly blur once coding begins.

## Why this document exists before any code

An extractor patched to match a target number is a dishonest instrument, not a correct one. Once a number exists to chase, the definition of success can quietly shift to fit whatever the pipeline produces, and the shift is invisible from inside the pipeline itself. This document fixes the problem statement and success criteria before any number exists, so that shift has nothing to attach to.

## The claim under test

> Given a model M, a context E, and a query X. The cache taken by slicing the E⊕X prefill at X's position, with absolute position correctly restored, carries information about E that is absent from the cache produced by prefilling X alone. This holds even though the two caches are identical in length and structure at the moment decoding begins.

This claim must be true or false independent of any particular model, dataset, or subject. If it only holds under one exact experimental setup, it is not a claim about a mechanism. It is an observation about one setup.

## Operational definition, not a target result

The slice under test is defined by the paper's Eq. 2:

```
C*(X) = C[|E| : |E|+|X|](E ⊕ X)
```

This is a definition of what is being tested, not a result to be matched. The absolute position of each token in this slice must still reflect its position in the full E⊕X prefill, not reset to zero. RoPE does not crash on a wrong position. It produces a wrong attention score with no signal at all, which is why this is stated as a rule rather than left to be caught later.

## Success criteria

There are three valid outcomes:

| Outcome | Condition | Meaning |
|---|---|---|
| **A. Gap present, correct direction** | Oracle > Direct, measured above noise | The claim holds at this scale and under these conditions |
| **B. Gap absent or reversed** | Oracle <= Direct, measured above noise | The claim does not hold under these conditions. This is a valid finding, not a failed replication |
| **C. Inconclusive** | The difference falls within the margin of uncertainty | No conclusion can yet be drawn. This is not an endpoint. It is a signal that measurement must be strengthened before any claim is made |

Outcome B is as valid a conclusion as outcome A. Reaching it with measurement that can be defended is a complete result. Scaling up to a larger model is a separate question and not a way to overturn outcome B.

## Why outcome C requires more than a point estimate

The effect under test may be small, a few percentage points. A single point estimate from one run cannot distinguish outcomes A, B, and C. A small difference may be a real signal or a sampling coincidence, depending on the variance underneath it. Because Direct and Oracle run on the exact same questions, this comparison is paired. One of the following two measures of uncertainty is required before A or B can be claimed:

- A bootstrap confidence interval over the paired accuracy difference, or
- A McNemar test over the questions answered correctly by one condition and incorrectly by the other

Without this, the reported result is a point estimate with no way to separate signal from coincidence.

## Two layers of ground truth that must not be mixed

A dataset's gold answer measures whether the model understood the question. That is not the same measurement as whether the extractor read the model's raw output correctly. The extractor is validated against a small, hand-labeled probe set, read by a human against raw text, before it is trusted to run on the full set. These two layers are validated separately, in this order, before any accuracy number is trusted.

## Scope

This document covers the single-model mechanism only: one model, its own cache, sliced and reused. It does not cover transfer between two different models, projection, or fusion. 

## Repository layout

```
cache-enrichment-first-principle/
├── cache_enrichment.py        # pure functions: slice_cache(), correct_position_ids()
├── extractor.py               # one parser per condition: Direct, Few-shot, Oracle
├── run_experiment.py          # wires the above together, writes raw per-sample output
├── analysis.py                # bootstrap CI and McNemar test over the paired accuracy
├── probe_labels.json          # hand-labeled ground truth used to validate the extractor
├── tests/
│   ├── test_slice_shape.py           # cache shape invariants after slicing
│   ├── test_position_correction.py   # regression test for the absolute-position bug
│   └── test_extractor_probe.py       # extractor precision against probe_labels.json
├── notebook/
│   └── walkthrough.ipynb      # narrated derivation, from claim to result
├── assets/
│   └── (finished GIFs and static figures referenced by the notebook)
├── results/
│   └── run_<date>.json        # raw per-sample output of each full run, kept for audit
└── README.md
```

`cache_enrichment.py` and `extractor.py` contain no I/O and no model loading. Both are testable in isolation, which is why `tests/` can check them without ever running a model.

## How to run

The steps are ordered. Each one gates the next.

1. **Set up the environment.**
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Label the probe set by hand**, before writing a single parser. Read raw model output for a small sample of each condition (Direct, Few-shot, Oracle) and record, by eye, what the model is actually answering. Save this as `probe_labels.json`. This step has no script; it exists to give `extractor.py` something honest to be checked against.

3. **Validate the extractor against the probe set.**
   ```
   python -m pytest tests/test_extractor_probe.py
   ```
   An extractor is not trusted on the full run until it passes this.

4. **Verify the cache mechanics in isolation**, independent of any extractor or dataset.
   ```
   python -m pytest tests/test_slice_shape.py tests/test_position_correction.py
   ```

5. **Run the full experiment.** See `run_experiment_guide.md` for the probe-first workflow and how to read the diagnostics.
   ```
   python run_experiment.py --out results/run_<date>.json
   ```

6. **Analyze the result.**
   ```
   python analysis.py results/run_<date>.json
   ```
   This reports the paired accuracy difference along with its bootstrap confidence interval and McNemar statistic, and states which of the three outcomes (A, B, or C) the run supports.

All scripts follow the same convention: silent and exit code 0 on success, a message on stderr and a non-zero exit code on failure. No script prints progress narration by default.