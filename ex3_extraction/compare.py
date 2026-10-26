"""Compare Step 4's variants from their saved runs: runs/{variant}_{n}.json.

Every variant is re-scored here with the same code, so a difference between two
variants comes from what changed between them, not from how they were scored.
An error that repeats in all runs is systematic; one that comes and goes is noise.
"""

import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ex3_extraction.ground_truth import load_ground_truth
from ex3_extraction.pipeline import RUNS_DIR, Outcome, Status, run_file, score_outcome
from ex3_extraction.scoring import FIELDS, DocScore, field_accuracy


def has_status(status: Status) -> Callable[[Outcome], bool]:
    return lambda outcome: outcome.status is status


def load_runs(runs_dir: Path, variant_name: str) -> list[list[Outcome]]:
    """Every saved run of one variant, in run order: runs/B_1.json, runs/B_2.json, ..."""
    runs = []
    run = 1
    while (path := run_file(runs_dir, variant_name, run)).exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        runs.append([Outcome.model_validate(o) for o in record["outcomes"]])
        run += 1
    if not runs:
        raise FileNotFoundError(f"no runs for variant {variant_name} in {runs_dir}")
    return runs


def compare(runs_dir: Path, variant_names: list[str]) -> str:
    truth = load_ground_truth()
    docs = sorted(truth)
    runs = {name: load_runs(runs_dir, name) for name in variant_names}
    scores: dict[str, list[DocScore]] = {
        name: [score_outcome(o, truth[o.doc].invoice) for run in variant_runs for o in run]
        for name, variant_runs in runs.items()
    }

    def per_doc(name: str, doc: str) -> list[DocScore]:
        return [score for score in scores[name] if score.doc == doc]

    width = 12
    header = "".join(name.rjust(width) for name in variant_names)
    lines = ["correct per document (summed over runs)", f"{'doc':<12}{header}  mismatches"]
    for doc in docs:
        cells = ""
        notes = []
        for name in variant_names:
            doc_scores = per_doc(name, doc)
            cells += f"{sum(s.correct for s in doc_scores)}/{len(doc_scores) * 12}".rjust(width)
            found = Counter(f"{m.field} ({m.kind})" for s in doc_scores for m in s.mismatches)
            if found:
                listed = ", ".join(f"{label} {n}/{len(doc_scores)}" for label, n in found.items())
                notes.append(f"{name}: {listed}")
        lines.append(f"{doc:<12}{cells}  {'; '.join(notes) or '-'}")

    totals = "".join(
        f"{sum(s.correct for s in scores[name])}/{len(scores[name]) * 12}".rjust(width)
        for name in variant_names
    )
    lines += [f"{'overall':<12}{totals}", "", f"{'field accuracy':<16}{header}"]
    accuracy = {name: field_accuracy(scores[name]) for name in variant_names}
    for field in FIELDS:
        cells = "".join(f"{accuracy[name][field]:.0%}".rjust(width) for name in variant_names)
        lines.append(f"{field:<16}{cells}")

    def stat(label: str, value: dict[str, str]) -> str:
        return f"{label:<26}" + "".join(value[name].rjust(width) for name in variant_names)

    outcomes = {name: [o for run in runs[name] for o in run] for name in variant_names}

    def count(test: Callable[[Outcome], bool]) -> dict[str, str]:
        return {n: str(sum(test(o) for o in outcomes[n])) for n in variant_names}

    def average(metric: Callable[[Outcome], int], digits: int) -> dict[str, str]:
        return {
            n: f"{sum(metric(o) for o in outcomes[n]) / len(outcomes[n]):.{digits}f}"
            for n in variant_names
        }

    lines += ["", f"{'pipeline':<26}{header}"]
    for status in Status:
        lines.append(stat(f"status {status.value}", count(has_status(status))))
    lines += [
        stat("changed_on_recheck", count(lambda o: bool(o.changed_on_recheck))),
        stat("calls per document", average(lambda o: len(o.attempts), 2)),
        stat(
            "input tokens per document",
            average(lambda o: sum(a.usage.input_tokens for a in o.attempts), 0),
        ),
        stat(
            "output tokens per document",
            average(lambda o: sum(a.usage.output_tokens for a in o.attempts), 0),
        ),
    ]
    lines.append(stat("runs", {n: str(len(runs[n])) for n in variant_names}))
    return "\n".join(lines)


if __name__ == "__main__":
    print(compare(RUNS_DIR, sys.argv[1:] or ["A", "B", "C"]))
