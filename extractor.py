"""Answer extraction, one parser per condition.

Pure functions. No I/O, no model loading.

The three parsers share a letter-matching core and differ only in where they
cut the raw output. That split is deliberate: the conditions do not disagree
about what an answer letter looks like, they disagree about where the model's
answer to the current question stops and something else begins. Encoding that
difference as three separate cut rules, rather than one regex that tries to be
clever, is what keeps each failure shape visible in the code.

Every parser returns a letter in {A, B, C, D} or None. None means the parser
could not find an answer, which is a result, not an error. It must never be
silently coerced into a wrong letter or a default.
"""

import re

VALID_LETTERS = ("A", "B", "C", "D")

# A letter that stands as an answer: at a word boundary, optionally followed by
# a separator that marks it as a label rather than a word. Matches "B", "B.",
# "B)", "(B)", "B:" but not the B inside "Because".
_ANSWER_LETTER = re.compile(
    r"(?:^|[^A-Za-z])\(?([ABCD])\)?(?=[\s.):,]|$)",
)


def _first_letter(text: str) -> str | None:
    """Return the first standalone answer letter in text, or None."""
    match = _ANSWER_LETTER.search(text)
    return match.group(1) if match else None


def _cut_at_first(text: str, markers: tuple[str, ...]) -> str:
    """Truncate text at the earliest occurrence of any marker."""
    cut = len(text)
    lowered = text.lower()
    for marker in markers:
        idx = lowered.find(marker.lower())
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def extract_direct(raw_output: str) -> str | None:
    """Parse Direct condition output.

    Failure shape: the model opens with prose or punctuation before stating
    the letter. "The correct answer is B." or "Answer: B". The letter is
    there, it just is not first.

    No truncation is needed. The first standalone letter in the output is the
    answer, because with no exemplar block there is nothing else in the text
    that could produce a stray letter. If the model refuses or rambles without
    ever committing to a letter, the answer is None, not a guess.
    """
    return _first_letter(raw_output)


# Markers that signal the model has stopped answering and started generating a
# new question, copying the shape of the exemplar block it was given.
_NEW_QUESTION_MARKERS = (
    "\nquestion",
    "\nq:",
    "\n\nquestion",
    "question:",
)


def extract_few_shot(raw_output: str) -> str | None:
    """Parse Few-shot condition output.

    Failure shape: the model answers correctly, then keeps going and invents
    a new question, carrying its own answer letter with it. A parser that
    reads the whole output can pick up a letter belonging to a question the
    model made up.

    Cut at the first new-question marker, then take the first letter in what
    remains. Everything after the cut is the model imitating the exemplar
    format, not answering.
    """
    answer_span = _cut_at_first(raw_output, _NEW_QUESTION_MARKERS)
    return _first_letter(answer_span)


def extract_oracle(raw_output: str) -> str | None:
    """Parse Oracle condition output.

    Failure shape: the model answers, then degenerates into a short repeating
    loop. Unlike the Few-shot case, the loop rarely introduces a competing
    letter, so the risk is lower. The cut is applied anyway rather than
    trusting that the loop stays harmless.

    Degeneration is detected structurally, by a line repeating, rather than by
    matching a fixed marker. A loop has no fixed vocabulary to match against.
    """
    answer_span = _cut_at_loop(raw_output)
    return _first_letter(answer_span)


def _cut_at_loop(text: str, min_repeats: int = 2) -> str:
    """Truncate text at the point a line starts repeating.

    Returns text unchanged if no line repeats min_repeats times.
    """
    lines = text.split("\n")
    seen: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        seen[stripped] = seen.get(stripped, 0) + 1
        if seen[stripped] >= min_repeats:
            return "\n".join(lines[:i])
    return text


EXTRACTORS = {
    "direct": extract_direct,
    "few_shot": extract_few_shot,
    "oracle": extract_oracle,
}