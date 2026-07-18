"""Score each parser in extractor.py against probe_labels.json.

probe_labels.json holds the raw model output for 20 hand-read probe samples
(tests/test_slice_shape.py and tests/test_position_correction.py cover
cache_enrichment.py; this file is extractor.py's counterpart) together with,
for each condition, the letter a human judged the model actually answered by
reading the raw text directly, per run_experiment_guide.md Step 2 - not the
extractor's own output. Scoring the parser against that independent label is
what makes this a test of the parser rather than a restatement of it.

This complements tests/test_extractor_synthetic.py rather than replacing it:
the synthetic file proves specific claims (a fixed gap, a fixed invariant)
with constructed strings; this file checks the parser against what the model
actually produced across a set deliberately chosen to include the adversarial
shapes seen so far (Direct hallucinating a new question, Few-shot
self-contradicting before hallucinating, samples where conditions genuinely
disagree on the answer).

PRECISION_THRESHOLD is 1.0, not a softer number, because n=20 per condition
is small enough that a single disagreement is a specific, inspectable case,
not noise to average away. If a future, larger hand-labeled set surfaces a
genuine unfixable disagreement, lowering this threshold should be a decision
made in the open, in this file, with the failing sample id named in a
comment, not a silent default.
"""

import json
import sys
from pathlib import Path

from extractor import EXTRACTORS

PRECISION_THRESHOLD = 1.0
LABELS_PATH = Path(__file__).parent.parent / "probe_labels.json"


def load_labels():
    with open(LABELS_PATH) as f:
        data = json.load(f)
    data.pop("_readme", None)
    return data


def score_condition(labels, condition):
    """Return (precision, mismatches) for one condition.

    mismatches is a list of (sample_id, expected, got) so a failure points
    directly at which sample to go read, not just a percentage.
    """
    fn = EXTRACTORS[condition]
    total = 0
    correct = 0
    mismatches = []
    for sid, entry in labels.items():
        raw = entry["raw"][condition]
        expected = entry["label"][condition]
        got = fn(raw)
        total += 1
        if got == expected:
            correct += 1
        else:
            mismatches.append((sid, expected, got))
    precision = correct / total if total else 0.0
    return precision, mismatches, total


def main():
    if not LABELS_PATH.exists():
        print(f"FAIL: {LABELS_PATH} not found. Hand-label a probe set before running this.", file=sys.stderr)
        return 1

    labels = load_labels()
    if not labels:
        print("FAIL: probe_labels.json has no samples.", file=sys.stderr)
        return 1

    overall_ok = True
    for condition in ("direct", "few_shot", "oracle"):
        precision, mismatches, total = score_condition(labels, condition)
        ok = precision >= PRECISION_THRESHOLD
        overall_ok = overall_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {condition}: {precision:.3f} precision over n={total} (threshold {PRECISION_THRESHOLD})")
        for sid, expected, got in mismatches:
            print(f"    sample {sid}: expected {expected!r}, got {got!r}")

    print()
    if overall_ok:
        print(f"all conditions meet the {PRECISION_THRESHOLD} threshold on n={len(labels)} hand-labeled samples")
        return 0
    else:
        print("at least one condition is below threshold; see mismatches above before running --n 100")
        return 1


if __name__ == "__main__":
    sys.exit(main())
