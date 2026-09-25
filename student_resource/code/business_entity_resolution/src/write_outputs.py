"""Write output/matching_results.tsv + output/candidate_pairs.tsv (the two
files utils/validate_submission.py and the scorer read), and run that
validator against them.

Format (enforced by validate_submission.py, reproduced here so a hole in the
pipeline can never silently produce a malformed file): TAB-separated, header
`source1_entity_id\\t{matched,candidate}_entity_ids`, exactly one row per
required Source-1 id (callers pass `required_ids` explicitly -- usually every
id in test_source1.tsv -- rather than this module inferring it, so a bug that
drops an S1 entity upstream shows up as a validator error, not a silently
short file), empty field for no match, comma-separated ids with no duplicates
within a row.
"""

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config


def _write_id_list_tsv(mapping: dict, required_ids, id_col: str, out_path: Path) -> None:
    """Shared writer for both matching_results.tsv and candidate_pairs.tsv.

    Inputs: {source1_entity_id: set(ids)}, the full set of ids that MUST get
    a row, the second column's header name, destination path.
    Output: none; writes the TSV.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", id_col])
        for s1id in sorted(required_ids):
            ids = mapping.get(s1id, set())
            writer.writerow([s1id, ",".join(sorted(ids))])


def write_matching_results(preds: dict, required_ids, out_path: Path) -> None:
    """Write the scored submission file: one row per required test S1 id,
    matched_entity_ids empty when the decision layer predicted no match.
    """
    _write_id_list_tsv(preds, required_ids, "matched_entity_ids", out_path)


def write_candidate_pairs(candidates: dict, required_ids, out_path: Path) -> None:
    """Write the (optional but expected) candidate-set file: one row per
    required test S1 id, candidate_entity_ids from blocking (pre-decision --
    a superset of matched_entity_ids for the same id).
    """
    _write_id_list_tsv(candidates, required_ids, "candidate_entity_ids", out_path)


def run_validator(matching_path: Path, candidate_path: Path, test_dir: Path) -> tuple:
    """Run utils/validate_submission.py against the two files just written.

    Runs it as a subprocess (its own docstring's documented usage) rather
    than importing it, so this always exercises exactly what a human running
    the same command from student_resource/ would see.

    Inputs: paths to the two output files, the test data dir (must contain
    test_source1.tsv; test_source2/3.tsv are only read with --check-ids,
    which this call does not pass -- see that script's docstring on why the
    ID-existence check is opt-in).
    Output: (returncode, combined stdout+stderr) -- 0 means safe to submit.
    """
    validator_path = config.REPO_ROOT / "utils" / "validate_submission.py"
    result = subprocess.run(
        [
            sys.executable, str(validator_path),
            "--matching", str(matching_path),
            "--candidate", str(candidate_path),
            "--test-dir", str(test_dir),
        ],
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout + result.stderr


def save_versioned_copy(matching_path: Path, candidate_path: Path, report: dict = None) -> Path:
    """Copy the two output files (+ an optional report.json) into
    submissions/<timestamp>/ for version history (the "keep version history"
    requirement from the working plan). submissions/ is gitignored (see
    .gitignore) -- these are generated artifacts, not source.
    """
    dest = config.SUBMISSIONS_DIR / time.strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(matching_path, dest / matching_path.name)
    if candidate_path.exists():
        shutil.copy(candidate_path, dest / candidate_path.name)
    if report is not None:
        with open(dest / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return dest
