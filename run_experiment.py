"""Run the three conditions over a dataset and write raw per-sample output.

Usage:
    python run_experiment.py --probe 2
    python run_experiment.py --n 100 --out results/run_2026-07-18.json

stdout carries the result. stderr carries diagnostics. Redirecting one does
not disturb the other:

    python run_experiment.py --n 100 --out results/run.json 2> run.log

Exit 0 on success. Non-zero and a message on stderr on failure, including the
abort gate firing.

Why there is a hand-written decode loop here instead of model.generate():

After the cache is sliced, two indices that generate() treats as one become
different. cache_position indexes into the cache, which holds len_X entries.
position_ids feed RoPE, and must start at len_E. For Direct and Few-shot these
agree, so the distinction never surfaces. For Oracle they diverge, and that
divergence is the entire thing under test. generate() derives position_ids from
cache_position internally, which would hand the variable under test to code that
does not know it is two variables. So the loop is written out.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter

import torch
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_enrichment import slice_cache
from extractor import EXTRACTORS

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 32
LETTERS = ("A", "B", "C", "D")
CONDITIONS = ("direct", "few_shot", "oracle")

# Questions come from MMLU-Redux, which re-annotated MMLU's ground truth.
# Exemplars come from the original MMLU dev split, which is where the standard
# 5-shot exemplars have always lived and which MMLU-Redux does not duplicate.
# Drawing exemplars from the same 100-question test pool would both shrink the
# evaluation set and leak questions into their own context.
QUESTION_DATASET = "edinburgh-dawg/mmlu-redux-2.0"
EXEMPLAR_DATASET = "cais/mmlu"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def format_question(row) -> str:
    choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"]))
    return f"Question: {row['question']}\n{choices}\nAnswer:"


def format_exemplars(rows) -> str:
    blocks = []
    for row in rows:
        answer = LETTERS[row["answer"]]
        blocks.append(f"{format_question(row)} {answer}")
    return "\n\n".join(blocks) + "\n\n"


@torch.no_grad()
def prefill(model, input_ids, position_ids):
    out = model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
    return out.logits, out.past_key_values


@torch.no_grad()
def decode_loop(model, tokenizer, cache, first_logits, start_position, max_new_tokens):
    """Greedy decode from an existing cache.

    Args:
        cache: cache to decode from. Its length may differ from start_position;
            that is the point.
        first_logits: logits at the last prefilled position, already computed.
        start_position: absolute position id of the first generated token. For
            Oracle this is len_E + len_X, not len_X.

    Returns:
        (generated_text, n_generated_tokens)
    """
    device = first_logits.device
    generated = []
    next_token = first_logits[0, -1, :].argmax(dim=-1).view(1, 1)

    for step in range(max_new_tokens):
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)

        position = torch.tensor([[start_position + step]], dtype=torch.long, device=device)
        cache_position = torch.tensor(
            [cache.get_seq_length()], dtype=torch.long, device=device
        )
        out = model(
            input_ids=next_token,
            position_ids=position,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        next_token = out.logits[0, -1, :].argmax(dim=-1).view(1, 1)

    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)


@torch.no_grad()
def run_sample(model, tokenizer, device, exemplar_text, question_text):
    """Run all three conditions on one sample.

    E ⊕ X is prefilled once and shared by Few-shot and Oracle. The two
    conditions differ in what they do with that cache, not in how it is built,
    so building it twice would burn the same compute for the same tensors.

    Order matters: the Oracle slice is taken before Few-shot decodes, because
    decoding appends to the cache in place. Slicing after would slice a cache
    that has grown by 32 tokens.

    Prefill and decode are timed separately. The efficiency claim is about
    decode, and Few-shot and Oracle share a prefill, so a single wall-clock
    number per condition would silently double-count it.
    """
    ids_E = tokenizer(exemplar_text, return_tensors="pt").input_ids.to(device)
    ids_X = tokenizer(question_text, return_tensors="pt").input_ids.to(device)
    ids_EX = torch.cat([ids_E, ids_X], dim=1)

    len_E = ids_E.shape[1]
    len_X = ids_X.shape[1]
    results = {}

    # Direct: prefill X alone. Positions start at zero, cache holds len_X.
    pos_direct = torch.arange(0, len_X, device=device).unsqueeze(0)
    t0 = time.perf_counter()
    logits_direct, cache_direct = prefill(model, ids_X, pos_direct)
    prefill_direct = time.perf_counter() - t0

    t0 = time.perf_counter()
    text, n_tok = decode_loop(model, tokenizer, cache_direct, logits_direct, len_X, MAX_NEW_TOKENS)
    results["direct"] = {
        "raw_output": text,
        "cache_tokens": len_X,
        "start_position": len_X,
        "generated_tokens": n_tok,
        "prefill_seconds": prefill_direct,
        "decode_seconds": time.perf_counter() - t0,
        "prefill_shared": False,
    }
    del cache_direct

    # One prefill of E ⊕ X, used by both Few-shot and Oracle.
    pos_few = torch.arange(0, len_E + len_X, device=device).unsqueeze(0)
    t0 = time.perf_counter()
    logits_ex, cache_ex = prefill(model, ids_EX, pos_few)
    prefill_ex = time.perf_counter() - t0

    # Oracle: discard E from the cache. Cache holds len_X, the same as Direct,
    # but every entry was computed while E was visible. Sliced first, because
    # Few-shot's decode will grow cache_ex in place.
    sliced = slice_cache(cache_ex, len_E, len_X)
    oracle_cache_len = sliced.get_seq_length()

    # Few-shot: keep the whole cache.
    t0 = time.perf_counter()
    text, n_tok = decode_loop(
        model, tokenizer, cache_ex, logits_ex, len_E + len_X, MAX_NEW_TOKENS
    )
    results["few_shot"] = {
        "raw_output": text,
        "cache_tokens": len_E + len_X,
        "start_position": len_E + len_X,
        "generated_tokens": n_tok,
        "prefill_seconds": prefill_ex,
        "decode_seconds": time.perf_counter() - t0,
        "prefill_shared": True,
    }
    del cache_ex

    t0 = time.perf_counter()
    text, n_tok = decode_loop(
        model, tokenizer, sliced, logits_ex, len_E + len_X, MAX_NEW_TOKENS
    )
    results["oracle"] = {
        "raw_output": text,
        "cache_tokens": oracle_cache_len,
        "start_position": len_E + len_X,
        "generated_tokens": n_tok,
        "prefill_seconds": prefill_ex,
        "decode_seconds": time.perf_counter() - t0,
        "prefill_shared": True,
    }

    return results, len_E, len_X


def log_sample_line(sample_id, len_E, len_X, conditions, extracted):
    """One line per sample.

    oracle_cache and oracle_pos are printed side by side on purpose. They must
    differ by len_E. If they are equal, the slice did not happen, and nothing
    downstream would reveal that.
    """
    oracle_cache = conditions["oracle"]["cache_tokens"]
    oracle_pos = conditions["oracle"]["start_position"]
    letters = " ".join(f"{c[:3]}={extracted[c] or '-'}" for c in CONDITIONS)
    log(
        f"[{sample_id:>4}] len_E={len_E:<4} len_X={len_X:<4} "
        f"oracle_cache={oracle_cache:<4} oracle_pos={oracle_pos:<4} {letters}"
    )


def log_parseable(counts, totals):
    rates = " ".join(
        f"{c[:3]}={counts[c] / totals[c]:.0%}" if totals[c] else f"{c[:3]}=n/a"
        for c in CONDITIONS
    )
    log(f"       parseable {rates}")


def log_summary(len_X_values, cache_lengths):
    """Final diagnostics.

    A cache length that never varies across samples means the slice arithmetic
    is not responding to the input. That signal is invisible in the accuracy
    numbers and only shows up here.
    """
    log("")
    log(
        f"len_X       min={min(len_X_values)} max={max(len_X_values)} "
        f"unique={len(set(len_X_values))}"
    )
    for cond in CONDITIONS:
        vals = cache_lengths[cond]
        log(f"cache {cond[:3]:<4} min={min(vals)} max={max(vals)} unique={len(set(vals))}")
    if len(set(cache_lengths["oracle"])) == 1:
        log("WARNING oracle cache length never varied, check slice arithmetic")


def resolve_subjects(dataset, spec):
    """Turn a subject spec into a list of subject names.

    'all' means every subject the dataset ships. A comma-separated list means
    exactly those.
    """
    if spec == "all":
        return sorted(get_dataset_config_names(dataset))
    return [s.strip() for s in spec.split(",") if s.strip()]


def load_questions(dataset, subjects):
    """Load the question pool across subjects, keeping only sound ground truth.

    MMLU-Redux exists because some of MMLU's gold answers are wrong. It ships
    both the sound and the unsound rows, labelled in error_type. Scoring
    against a gold answer that the dataset's own annotators marked as wrong
    measures the dataset, not the model.

    This filter matters more than it looks. A bad gold answer penalises all
    three conditions equally, so it never shows up as an anomaly in any single
    number, and it never shows up as a gap between conditions either. It just
    quietly pushes every accuracy down and adds noise to the paired difference.

    Each kept row is tagged with its subject, because the exemplars it gets
    must come from the same subject.

    Returns:
        (rows, n_total, dropped_counter)
    """
    kept = []
    dropped = Counter()
    n_total = 0
    for subject in subjects:
        ds = load_dataset(dataset, subject, split="test")
        n_total += len(ds)
        for row in ds:
            if row.get("error_type") == "ok":
                kept.append({**row, "subject": subject})
            else:
                dropped[row.get("error_type")] += 1
    return kept, n_total, dropped


def load_exemplars(dataset, subjects, n_shots):
    """Load exemplars from the MMLU dev split, one block per subject.

    The dev split holds exactly the questions MMLU intended as few-shot
    exemplars. Asking for more than it holds is a configuration error, not
    something to paper over by borrowing from the test split.

    Exemplars are kept per subject. An anatomy question shown virology
    exemplars is not few-shot prompting, it is noise with a longer prefix.

    Returns:
        {subject: exemplar_text}
    """
    out = {}
    for subject in subjects:
        ds = load_dataset(dataset, subject, split="dev")
        if len(ds) < n_shots:
            raise ValueError(f"{subject}: dev split has {len(ds)} rows, need {n_shots} shots")
        out[subject] = format_exemplars([ds[i] for i in range(n_shots)])
    return out


def probe(model, tokenizer, device, rows, exemplars, n_probe):
    """Read raw output by eye before committing to a full run.

    Prints everything, writes nothing. This is not logging, it is a reading
    tool. The full run is not worth starting until this looks sane.
    """
    for i, row in enumerate(rows[:n_probe]):
        conditions, len_E, len_X = run_sample(
            model, tokenizer, device, exemplars[row["subject"]], format_question(row)
        )
        log("=" * 70)
        log(
            f"sample {i}  subject={row['subject']}  gold={LETTERS[row['answer']]}  "
            f"len_E={len_E}  len_X={len_X}"
        )
        for cond in CONDITIONS:
            payload = conditions[cond]
            extracted = EXTRACTORS[cond](payload["raw_output"])
            log("-" * 70)
            log(f"{cond}  cache={payload['cache_tokens']}  start_pos={payload['start_position']}")
            log(f"raw: {payload['raw_output']!r}")
            log(f"extracted: {extracted}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        help="number of samples for a full run. Defaults to every sound row available.",
    )
    parser.add_argument("--out", type=str, help="output path for a full run")
    parser.add_argument("--probe", type=int, help="probe mode: read N samples, write nothing")
    parser.add_argument("--dataset", type=str, default=QUESTION_DATASET)
    parser.add_argument("--exemplar-dataset", type=str, default=EXEMPLAR_DATASET)
    parser.add_argument(
        "--subject",
        type=str,
        default="all",
        help="'all', or a comma-separated list of subjects",
    )
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gate-after",
        type=int,
        default=10,
        help="check the abort gate after this many samples",
    )
    parser.add_argument(
        "--gate-min-parseable",
        type=float,
        default=0.5,
        help=(
            "abort if any condition's parseable rate is below this at the gate. "
            "This default is a guess, not a measurement. Set it from probe output."
        ),
    )
    args = parser.parse_args()

    if args.probe is None and args.out is None:
        print("need either --probe N, or --out PATH", file=sys.stderr)
        return 2

    device = "cpu"
    torch.manual_seed(args.seed)

    subjects = resolve_subjects(args.dataset, args.subject)
    log(f"subjects: {len(subjects)}")

    questions, n_total, dropped = load_questions(args.dataset, subjects)
    log(f"{n_total} rows, {len(questions)} sound, {sum(dropped.values())} dropped")
    for error_type, count in dropped.most_common():
        log(f"  dropped {error_type}: {count}")
    if not questions:
        print("no sound rows after filtering error_type", file=sys.stderr)
        return 1

    try:
        exemplars = load_exemplars(args.exemplar_dataset, subjects, args.n_shots)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Shuffle before truncating. Without this, --n takes a prefix, which is one
    # subject's worth of questions, not a sample of the pool. The abort gate
    # reads the first samples too, and would judge the extractor on one subject.
    rng = random.Random(args.seed)
    rng.shuffle(questions)

    n_run = args.n if args.n is not None else len(questions)
    if n_run > len(questions):
        print(f"asked for {n_run} samples, only {len(questions)} sound rows", file=sys.stderr)
        return 1
    questions = questions[:n_run]
    log(f"running {len(questions)} samples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.to(device)
    model.eval()

    if args.probe is not None:
        return probe(model, tokenizer, device, questions, exemplars, args.probe)

    rows = []
    parsed_counts = Counter()
    totals = Counter()
    len_X_values = []
    cache_lengths = {c: [] for c in CONDITIONS}

    for n_done, (i, row) in enumerate(enumerate(questions), start=1):
        conditions, len_E, len_X = run_sample(
            model, tokenizer, device, exemplars[row["subject"]], format_question(row)
        )

        extracted = {}
        for cond in CONDITIONS:
            payload = conditions[cond]
            letter = EXTRACTORS[cond](payload["raw_output"])
            extracted[cond] = letter
            totals[cond] += 1
            if letter is not None:
                parsed_counts[cond] += 1
            cache_lengths[cond].append(payload["cache_tokens"])
            rows.append(
                {
                    "sample_id": i,
                    "subject": row["subject"],
                    "condition": cond,
                    "gold": LETTERS[row["answer"]],
                    "len_E": len_E,
                    "len_X": len_X,
                    "extracted": letter,
                    **payload,
                }
            )
        len_X_values.append(len_X)

        log_sample_line(i, len_E, len_X, conditions, extracted)
        log_parseable(parsed_counts, totals)

        if n_done == args.gate_after:
            failing = {
                c: parsed_counts[c] / totals[c]
                for c in CONDITIONS
                if parsed_counts[c] / totals[c] < args.gate_min_parseable
            }
            if failing:
                log("")
                log(f"ABORT at sample {n_done}: parseable rate below gate")
                for cond, rate in failing.items():
                    log(f"  {cond} {rate:.0%} < {args.gate_min_parseable:.0%}")
                log("nothing written. read the raw output with --probe before rerunning.")
                return 1

    log_summary(len_X_values, cache_lengths)

    with open(args.out, "w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "dataset": args.dataset,
                "exemplar_dataset": args.exemplar_dataset,
                "subjects": subjects,
                "n": len(questions),
                "n_total_before_filter": n_total,
                "dropped_error_types": dict(dropped),
                "n_shots": args.n_shots,
                "seed": args.seed,
                "rows": rows,
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())