"""Hand-crafted adversarial and control cases for extractor.py.

This file is deliberately separate from test_extractor_probe.py. That file
scores parsers against probe_labels.json - real model output, hand-labeled by
reading raw text. This file exists for shapes the 20-sample probe never
happened to produce, but that are already named as known risks in
run_experiment_guide.md's anomaly table. A probe of any finite size can miss a
failure shape by luck; a synthetic case makes the claim about that shape
checkable regardless of what the model happened to say on a given day.

Each case is asserted directly, not scored against a threshold. A synthetic
case failing is not noise to average away - it is a concrete, actionable gap
in the parser.
"""

from extractor import extract_direct, extract_few_shot, extract_oracle


def check(label, fn, raw, expected, note=""):
    got = fn(raw)
    status = "PASS" if got == expected else "FAIL"
    print(f"[{status}] {label}: expected={expected!r} got={got!r}" + (f"  ({note})" if note else ""))
    return status == "PASS"


results = []

# --- extract_direct: control cases (should already pass) ---

results.append(check(
    "direct_control_leading_prose",
    extract_direct,
    "The answer is B.",
    "B",
    "leading prose before the letter, no restated choice list",
))

results.append(check(
    "direct_control_no_false_hit_in_word",
    extract_direct,
    "Because the reaction is exothermic, B is correct.",
    "B",
    "'Because' contains a B that must not match as a standalone letter",
))

results.append(check(
    "direct_control_answer_then_explanation",
    extract_direct,
    " B\n\n**Step-by-Step Explanation:**\nThe reaction proceeds because...",
    "B",
    "matches the common clean shape seen in the 20-sample probe",
))

# --- extract_direct: the identified gap ---

results.append(check(
    "direct_GAP_restates_choice_list",
    extract_direct,
    "A. Paris\nB. London\nC. Berlin\nD. Madrid\nThe correct answer is B.",
    "B",
    "KNOWN RISK from run_experiment_guide.md anomaly table: 'extract_direct is "
    "known to do this when the model restates the choice list before answering'. "
    "None of the 20 probed Direct samples happened to restate the choice list, "
    "so this shape was invisible to the probe alone. Fixed by _skip_choice_list.",
))

results.append(check(
    "direct_control_single_line_answer_in_choice_format_not_skipped",
    extract_direct,
    "B. This is because the reaction is exothermic.",
    "B",
    "REGRESSION GUARD for the fix above: a single line in choice-format is "
    "the model's real answer, not a restated list, and min_lines=2 must not "
    "skip it.",
))

results.append(check(
    "direct_control_partial_restate_then_answer",
    extract_direct,
    "A. Paris\nB. London\nAnswer: B",
    "B",
    "Two restated choices (not all four) still get skipped; the search "
    "resumes at 'Answer: B' once a non-choice-shaped line appears.",
))

# --- extract_oracle: control cases matching what the probe actually saw ---

results.append(check(
    "oracle_control_line_repeat_cut",
    extract_oracle,
    "B\n**\n**\n**\n**\n",
    "B",
    "matches sample 3/8 shape: identical repeating line, cut engages",
))

results.append(check(
    "oracle_control_same_letter_loop",
    extract_oracle,
    "A\n\nA\n\nA\n\nA\n\n",
    "A",
    "matches sample 7 shape: the loop repeats the answer letter itself",
))

# --- extract_oracle: the identified gap ---

results.append(check(
    "oracle_INVARIANT_same_line_loop_harmless_by_construction",
    extract_oracle,
    "C\n\nrepeatfragmentrepeatfragmentrepeatfragmentD repeatfragment",
    "C",
    "_cut_at_loop only detects repetition across separate lines, so a "
    "same-line degenerate loop (like 'osten...', 'polers polers...' seen in "
    "the real probe) is never cut. This is harmless, not a blind spot: "
    "tests/test_decode_loop_first_token_invariant.py proves decode_loop "
    "appends argmax(first_logits) as token 0 before the loop body ever runs, "
    "so no downstream degeneration - same-line or not - can ever precede the "
    "answer letter for Oracle or Few-shot. No extractor change needed here.",
))

results.append(check(
    "oracle_INVARIANT_would_only_break_if_decode_loop_changed",
    extract_oracle,
    "repeat fragment repeat fragment D repeat fragment\nC",
    "D",
    "This case is constructed to fail on purpose: it simulates degenerate "
    "content appearing BEFORE the answer, which decode_loop's structure "
    "(see test_decode_loop_first_token_invariant.py) makes impossible today. "
    "It exists as a tripwire: if this string were ever built from real model "
    "output, extract_oracle would silently return the wrong letter, so any "
    "future change to decode_loop's control flow must be checked against "
    "the invariant test, not assumed safe.",
))

# --- extract_few_shot: control case exercising the multi-letter cut span (sample 4 shape) ---

results.append(check(
    "few_shot_control_self_contradiction_before_hallucination",
    extract_few_shot,
    " C\nAnswer:\nA\n\nQuestion: Which of the following statements best captures...",
    "C",
    "matches sample 4: model states C, then contradicts with 'Answer:\\nA', "
    "then hallucinates a new question. Cut removes the hallucinated question "
    "block; leftmost letter in what remains (C) wins over the later A.",
))

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} passed")
