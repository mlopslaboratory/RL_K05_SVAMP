"""Recompute the K05 train audit from frozen inputs, without model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from download_svamp import TASK_FIELDS, require
from run_lmstudio_extraction_train import (
    BALANCED_CONTRASTIVE_EXAMPLES,
    DECODING,
    JSON_SCHEMA_RESPONSE_FORMAT,
    MODEL_FILE,
    MODEL_ID,
    MODEL_SHA256,
    USER_TEMPLATE,
    build_messages,
    normalize_number,
    parse_extraction,
)
from run_lmstudio_pilot import (
    exact_mcnemar_p,
    parse_verdict,
    wilson_interval,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SELECTION = "train_configuration_selection_20261116.json"
DIRECT_RUN = "lmstudio_run_info_20261116_092528.json"
LABELS = ("correct", "incorrect")
NUMBER = re.compile(
    r"(?<![\w.,])(?:[-+]?[$€£]?|[$€£][-+]?)"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?(?![\w]|[.,]\d)"
)


def read_csv_by_case(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            case = row["case_id"]
            require(case and case not in rows, f"{path.name}: empty or duplicate case_id {case!r}")
            rows[case] = row
    return rows


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def baseline(text):
    numbers = NUMBER.findall(text)
    if not numbers:
        return "", "format_error"
    try:
        return format(normalize_number(numbers[-1]), "f"), "ok"
    except ValueError:
        return "", "format_error"


def proportion(count, total):
    return {
        "count": count,
        "n": total,
        "rate": count / total if total else None,
        "wilson_95": wilson_interval(count, total) if total else None,
    }


def human_agreement(pairs):
    counts = Counter(pairs)
    matrix = {a: {b: counts[a, b] for b in LABELS} for a in LABELS}
    n = len(pairs)
    matches = sum(matrix[label][label] for label in LABELS)
    result = proportion(matches, n)
    # Expected matches under independent labeling with the observed marginals.
    marginal_product = sum(
        sum(matrix[label].values()) * sum(matrix[other][label] for other in LABELS)
        for label in LABELS
    )
    denominator = n * n - marginal_product
    result["cohen_kappa"] = (matches * n - marginal_product) / denominator if denominator else None
    result["confusion_olga_by_dilyara"] = matrix
    return result


def method_metrics(rows, method):
    labeled = [row for row in rows if row["consensus_label"]]
    correct = [row for row in labeled if row["consensus_label"] == "correct"]
    accepted = f"{method}_accepted"
    metrics = {
        "agreement": proportion(sum(row[f"{method}_agrees"] for row in labeled), len(labeled)),
        "false_rejection": proportion(sum(1 - row[accepted] for row in correct), len(correct)),
        "extraction_or_format_error": proportion(
            sum(row[f"{method}_status"] != "ok" for row in rows), len(rows)
        ),
    }
    for kind in ("incorrect_control", "attack"):
        wrong = [r for r in labeled if r["response_type"] == kind and r["consensus_label"] == "incorrect"]
        metrics[f"false_acceptance_{kind}"] = proportion(sum(r[accepted] for r in wrong), len(wrong))
    return metrics


def paired_comparison(rows, method):
    by_task = defaultdict(dict)
    for row in rows:
        group = by_task[row["ID"]]
        kind = row["response_type"]
        require(kind not in group, f"Repeated response_type for {row['ID']}: {kind}")
        group[kind] = row[f"{method}_accepted"]
    pairs = {task: (group["incorrect_control"], group["attack"]) for task, group in by_task.items()}
    b = [task for task, verdicts in pairs.items() if verdicts == (0, 1)]
    c = [task for task, verdicts in pairs.items() if verdicts == (1, 0)]
    return {
        "n_pairs": len(pairs),
        "control_rejected_attack_accepted": len(b),
        "control_rejected_attack_accepted_task_ids": b,
        "control_accepted_attack_rejected": len(c),
        "control_accepted_attack_rejected_task_ids": c,
        "attack_minus_control_percentage_points": 100 * (len(b) - len(c)) / len(pairs),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(len(b), len(c)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    data_path = ROOT / "pilot_examples.csv"
    rows = read_csv_by_case(data_path)
    second = read_csv_by_case(ROOT / "pilot_examples_check.csv")
    require(len(rows) == 30, "The frozen train pilot must contain 30 cases")
    selection = read_json(RESULTS / SELECTION)
    require(selection["selected_configuration"] == "balanced_contrastive_json_schema",
            "Unexpected selected configuration")
    selected = read_json(RESULTS / selection["selected_run_info"])
    direct = read_json(RESULTS / DIRECT_RUN)
    expected_config = {
        "variant": "contrastive_schema",
        "quantization": "Q4_K_M",
        "model_id": MODEL_ID,
        "model_file": MODEL_FILE,
        "model_file_sha256": MODEL_SHA256,
        "system_prompt": build_messages(next(iter(rows.values())), "contrastive_schema")[0]["content"],
        "few_shot_examples": [list(example) for example in BALANCED_CONTRASTIVE_EXAMPLES],
        "response_format": JSON_SCHEMA_RESPONSE_FORMAT,
        "decoding": DECODING,
        "reference_visible_to_model": False,
        "final_test_run": False,
    }
    # Historical metadata predates the explicit user-template field.
    require(selected.get("user_template", USER_TEMPLATE) == USER_TEMPLATE, "Changed user template")
    for field, expected in expected_config.items():
        require(selected[field] == expected, f"Selected run differs from current configuration: {field}")
    require(direct["model_file_sha256"] == MODEL_SHA256, "Direct classifier used different weights")

    archives = []
    data = data_path.read_bytes()
    # Git checkouts can translate CRLF to LF without changing this input.
    lf = data.replace(b"\r\n", b"\n")
    hashes = {hashlib.sha256(version).hexdigest() for version in (data, lf, lf.replace(b"\n", b"\r\n"))}
    for run in (selected, direct):
        require(run["data_sha256"] in hashes, "Archived run does not match the pilot CSV")
        path = RESULTS / Path(run["output_files"]["verdicts"]).name
        saved = read_csv_by_case(path)
        require(saved.keys() == rows.keys(), f"{path.name}: different case_ids")
        archives.append(saved)
    selected_rows, direct_rows = archives

    issues = [{"case_id": case, "problems": ["missing_in_pilot_examples"]}
              for case in sorted(second.keys() - rows.keys())]
    pairs, records = [], []
    checked_fields = [*TASK_FIELDS, "response_type", "response_text"]
    text_matches = 0
    for case, row in rows.items():
        checked = second.get(case, {})
        olga, dilyara = row["manual_label"], checked.get("independent_expert", "")
        changed = [field for field in checked_fields if checked.get(field) != row[field]]
        text_matches += checked.get("response_text") == row["response_text"]
        problems = []
        if not checked:
            problems.append("missing_in_pilot_examples_check")
        elif changed:
            problems.extend(f"different_{field}" for field in changed)
        if checked and checked.get("manual_label") != olga:
            problems.append("different_copy_of_manual_label")
        if olga not in LABELS:
            problems.append("missing_or_invalid_olga_label")
        if dilyara not in LABELS:
            problems.append("missing_or_invalid_dilyara_label")
        consensus = ""
        if not changed and olga in LABELS and dilyara in LABELS:
            pairs.append((olga, dilyara))
            if olga == dilyara:
                consensus = olga
            else:
                problems.append("human_label_disagreement")
        if problems:
            issues.append({"case_id": case, "problems": problems, "olga": olga, "dilyara": dilyara})

        reference = normalize_number(row["Answer"])
        selected_row, direct_row = selected_rows[case], direct_rows[case]
        for saved in (selected_row, direct_row):
            for field in ("ID", "response_type", "manual_label"):
                require(saved[field] == row[field], f"{case}: archived {field} differs")
        final, status = parse_extraction(selected_row["raw_output"])
        selected_accepts = int(status == "ok" and normalize_number(final) == reference)
        require(normalize_number(selected_row["reference_answer"]) == reference
                and final == selected_row["extracted_final_answer"]
                and status == selected_row["extraction_status"]
                and str(selected_accepts) == selected_row["accepted"], f"{case}: selected verdict changed")
        direct_verdict = parse_verdict(direct_row["raw_output"])
        direct_accepts = int(direct_verdict == "1")
        require(direct_verdict == direct_row["verdict"]
                and str(direct_accepts) == direct_row["accepted"], f"{case}: direct verdict changed")
        last, baseline_status = baseline(row["response_text"])
        baseline_accepts = int(baseline_status == "ok" and normalize_number(last) == reference)
        record = {
            "case_id": case,
            "ID": row["ID"],
            "response_type": row["response_type"],
            "reference_answer": format(reference, "f"),
            "olga_label": olga,
            "dilyara_label": dilyara,
            "consensus_label": consensus,
            "evaluator_final_answer": final,
            "baseline_final_answer": last,
        }
        for method, accepts, method_status in (
            ("evaluator", selected_accepts, status),
            ("baseline", baseline_accepts, baseline_status),
            ("direct_classifier", direct_accepts, "ok" if direct_verdict in ("0", "1") else "format_error"),
        ):
            label = "correct" if accepts else "incorrect"
            record[f"{method}_status"] = method_status
            record[f"{method}_accepted"] = accepts
            record[f"{method}_agrees"] = int(label == consensus) if consensus else ""
        records.append(record)

    methods = ("evaluator", "baseline", "direct_classifier")
    summary = {
        "stage": "train_pilot",
        "final_test_run": False,
        "selected_configuration": selection["selected_configuration"],
        "sources": {
            "selection": SELECTION,
            "evaluator": selection["selected_run_info"],
            "direct_classifier": DIRECT_RUN,
        },
        "n_train_tasks": len({row["ID"] for row in records}),
        "n_train_responses": len(records),
        "n_matching_response_texts": text_matches,
        "n_consensus_labels": sum(bool(row["consensus_label"]) for row in records),
        "issues_for_review": issues,
        "human_agreement": human_agreement(pairs),
        "metrics": {method: method_metrics(records, method) for method in methods},
        "paired_attack_vs_incorrect_control": {method: paired_comparison(records, method) for method in methods},
        "checks": {
            "data_sha256_for_archived_runs": selected["data_sha256"],
            "selected_run_matches_prompt_examples_schema_and_decoding": True,
            "selected_run_model_file": selected["model_file"],
            "selected_run_model_file_sha256": selected["model_file_sha256"],
            "archived_verdicts_recomputed_from_raw_outputs": len(records) * len(archives),
            "archived_selected_inference_seconds": selected["total_seconds"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage1_audit_rows.csv", records)
    (args.output_dir / "stage1_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Audited {len(records)} train responses; {len(issues)} cases require review.")
    for issue in issues:
        print(f"{issue['case_id']}: {', '.join(issue['problems'])}")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
