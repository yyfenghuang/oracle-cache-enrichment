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
    "oracle_INVARIANT_would_fail_safe_if_decode_loop_changed",
    extract_oracle,
    "repeat fragment repeat fragment D repeat fragment\nC",
    None,
    "Constructed to simulate degenerate content appearing BEFORE the answer, "
    "which decode_loop's structure (test_decode_loop_first_token_invariant.py) "
    "makes impossible today. Under the old _first_letter (leftmost regex "
    "match), this case silently returned the wrong letter 'D'. Under "
    "_first_char_letter (checks only the literal first character), it "
    "returns None instead: a safe 'no answer found' rather than a "
    "confidently wrong one. If this string were ever built from real model "
    "output, a None here should prompt going back to the probe, not be "
    "mistaken for a crash. Kept as a tripwire for decode_loop control-flow "
    "changes either way.",
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

# --- extract_oracle / extract_few_shot: letter glued directly to degenerate
#     continuation with no boundary character, found in results/run_2026-07-19_n2000.json ---

results.append(check(
    "oracle_GAP_letter_glued_to_word_no_boundary",
    extract_oracle,
    " CANNOT LIMIT THE INFRACANANANANANANANANANANANANANANANANANANAN",
    "C",
    "REAL DATA, sample 412 of the n=2000 run. Degenerate continuation glued "
    "directly onto the answer letter with zero separator, forming what reads "
    "as the word 'CANNOT'. _first_letter's boundary requirement missed this "
    "entirely (returned None on a clearly-answered sample). Fixed by "
    "_first_char_letter, justified by the same first-token invariant as "
    "above.",
))

results.append(check(
    "oracle_GAP_letter_glued_to_backtick_loop",
    extract_oracle,
    " C`.\n\n`.`.`.`.`.`.`.`.`.`.`.`.`.",
    "C",
    "REAL DATA, sample 421 of the n=2000 run. Backtick is not in "
    "_first_letter's boundary character class, so 'C`.' was missed too.",
))

results.append(check(
    "oracle_GAP_letter_glued_to_malformed_unicode",
    extract_oracle,
    " B\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\u00b3\u00b3\u00b3",
    "B",
    "REAL DATA, sample 495 of the n=2000 run. Malformed decode bytes glued "
    "onto the letter with no boundary character.",
))

# --- extract_direct: \boxed{LETTER} notation, found in results/run_2026-07-19_n2000.json ---

results.append(check(
    "direct_GAP_boxed_letter_notation",
    extract_direct,
    " \\boxed{C}\nAnswer:\nTo solve the problem, we are given the operation...",
    "C",
    "REAL DATA, sample 1350 of the n=2000 run. Math-reasoning models "
    "sometimes answer in LaTeX \\boxed{} notation. '}' is not a boundary "
    "character _first_letter accepts, so the letter was missed even though "
    "unambiguous. Fixed by a _BOXED_LETTER fallback tried only after the "
    "primary letter search fails.",
))

results.append(check(
    "direct_control_boxed_non_letter_stays_none",
    extract_direct,
    " 46\nSo, the answer is \\boxed{46}\nAnswer: \\boxed{46}",
    None,
    "REAL DATA, sample 1510 of the n=2000 run. A boxed NUMBER is not a "
    "mis-parsed letter - Direct has no obligation to answer in letter form, "
    "and this None is correct, not a gap. Regression guard: the boxed "
    "fallback must not turn this into a wrong letter.",
))

results.append(check(
    "direct_control_truncated_reasoning_stays_none",
    extract_direct,
    " Let's see. The question is asking about the world history event that "
    "caused the Soviet Union to enter Korea. The passage mentions that "
    "Bonesteel was considering the",
    None,
    "REAL DATA, sample 380 of the n=2000 run. Reasoning cut off by the "
    "32-token budget before stating a letter. This None is correct: no "
    "answer was ever produced, this is not an extraction failure.",
))

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} passed")
