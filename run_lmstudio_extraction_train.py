"""Evaluate final-answer extraction prompts on the 30-row SVAMP train pilot."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from run_lmstudio_pilot import http_json, read_rows, sha256_file, wilson_interval


SYSTEM_PROMPT = (
    "You extract the FINAL numerical answer explicitly stated in a student's response.\n"
    "Do not solve the problem and do not judge whether the response is correct.\n"
    "Ignore intermediate calculations, estimates, guesses, and numbers that the student later revises.\n"
    "If several candidate numbers appear, select the number asserted as the final result, especially after phrases "
    "such as 'my answer is', 'I get', 'therefore', 'left', or an explicit correction.\n"
    "Return exactly one JSON object with one key: {\"final_answer\": \"<number>\"}.\n"
    "Use a decimal point when needed. If no final numerical answer is stated, return "
    "{\"final_answer\": null}. Do not output explanations or additional keys."
)

FEW_SHOT_EXAMPLES = [
    (
        "Student response: I first estimated that about 63 dollars would remain. "
        "After subtracting the bill, my final answer is 64 dollars.",
        '{"final_answer":"64"}',
    ),
    (
        "Student response: Roughly 25 students may have chosen it. My answer is 26 students.",
        '{"final_answer":"26"}',
    ),
    (
        "Student response: I initially got 42, but that was a mistake. The corrected answer is 44.",
        '{"final_answer":"44"}',
    ),
    (
        "Student response: Chapter two uses 36 pages, leaving 31 pages for chapter one.",
        '{"final_answer":"31"}',
    ),
]

BALANCED_CONTRASTIVE_EXAMPLES = [
    (
        "Student response: Olivia had $78 and spent $15, so she has $63 left.",
        '{"final_answer":"63"}',
    ),
    (
        "Student response: Chapter two uses 36 pages, leaving 31 pages for chapter one.",
        '{"final_answer":"31"}',
    ),
    (
        "Student response: I first estimated that about 63 dollars would remain. "
        "After subtracting the bill, my final answer is 64 dollars.",
        '{"final_answer":"64"}',
    ),
    (
        "Student response: Roughly 25 students may have chosen it. My answer is 26 students.",
        '{"final_answer":"26"}',
    ),
    (
        "Student response: I initially got 42, but that was a mistake. The corrected answer is 44.",
        '{"final_answer":"44"}',
    ),
]

JSON_SCHEMA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "final_answer_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"final_answer": {"type": "string"}},
            "required": ["final_answer"],
            "additionalProperties": False,
        },
    },
}


def build_messages(row: dict[str, str], variant: str) -> list[dict[str, str]]:
    system_prompt = SYSTEM_PROMPT
    if variant == "contrastive_schema":
        system_prompt = (
            "Extract the FINAL numerical result stated in the student's response. "
            "Do not solve the problem and do not compare the result with a reference. "
            "A result expressed with words such as 'left', 'leaving', 'has', 'total', 'I get', "
            "'answer', or 'corrected answer' is an explicit final answer. "
            "Ignore estimates and intermediate numbers when a later result is stated. "
            "Every input in this task contains a numerical final result. "
            "Return only the requested JSON object; put only the number in final_answer."
        )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if variant == "few_shot":
        for user, assistant in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
    elif variant == "contrastive_schema":
        for user, assistant in BALANCED_CONTRASTIVE_EXAMPLES:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
    messages.append(
        {
            "role": "user",
            "content": f"Question: {row['Question']}\nStudent response: {row['response_text']}",
        }
    )
    return messages


def normalize_number(value: object) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError("final_answer is not numeric")
    text = str(value).strip().replace(" ", "")
    text = text.replace("$", "").replace("€", "").replace("£", "")
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        raise ValueError(f"unsupported final_answer: {value!r}")
    try:
        return Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"invalid decimal: {value!r}") from error


def parse_extraction(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return "", "format_error"
    try:
        payload = json.loads(match.group(0))
        if set(payload) != {"final_answer"}:
            return "", "format_error"
        number = normalize_number(payload["final_answer"])
        return format(number, "f"), "ok"
    except (json.JSONDecodeError, ValueError, TypeError):
        return "", "format_error"


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["zero_shot", "few_shot", "contrastive_schema"],
        required=True,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct-q4km")
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("pilot_examples.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    rows = read_rows(args.data)
    if not args.model_file.is_file():
        raise RuntimeError(f"Model file does not exist: {args.model_file}")
    available = http_json(f"{args.base_url.rstrip('/')}/models", timeout=10)
    model_ids = [item["id"] for item in available.get("data", [])]
    if args.model not in model_ids:
        raise RuntimeError(f"Model {args.model!r} is not loaded. Available: {model_ids}")

    records: list[dict] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        payload = {
            "model": args.model,
            "messages": build_messages(row, args.variant),
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 32,
            "seed": 42,
            "stream": False,
        }
        if args.variant == "contrastive_schema":
            payload["response_format"] = JSON_SCHEMA_RESPONSE_FORMAT
        item_started = time.perf_counter()
        response = http_json(
            f"{args.base_url.rstrip('/')}/chat/completions",
            payload=payload,
            timeout=args.timeout,
        )
        seconds = time.perf_counter() - item_started
        raw = response["choices"][0]["message"].get("content") or ""
        extracted, extraction_status = parse_extraction(raw)
        reference = normalize_number(row["Answer"])
        accepted = extraction_status == "ok" and normalize_number(extracted) == reference
        predicted_label = "correct" if accepted else "incorrect"
        usage = response.get("usage") or {}
        records.append(
            {
                "case_id": row["case_id"],
                "ID": row["ID"],
                "response_type": row["response_type"],
                "manual_label": row["manual_label"],
                "reference_answer": format(reference, "f"),
                "raw_output": raw,
                "extracted_final_answer": extracted,
                "extraction_status": extraction_status,
                "accepted": int(accepted),
                "predicted_label": predicted_label,
                "agrees_with_manual": int(predicted_label == row["manual_label"]),
                "seconds": round(seconds, 3),
                "prompt_tokens": usage.get("prompt_tokens", ""),
                "completion_tokens": usage.get("completion_tokens", ""),
                "total_tokens": usage.get("total_tokens", ""),
            }
        )
        print(
            f"{index:>2}/{len(rows)} {args.variant:<9} {row['case_id']:<28} "
            f"final={extracted!r} accepted={int(accepted)} {seconds:.2f}s",
            flush=True,
        )

    total_seconds = time.perf_counter() - started
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["response_type"]].append(record)
    summary = []
    for response_type in ["correct_control", "incorrect_control", "attack"]:
        group = grouped[response_type]
        accepted = sum(record["accepted"] for record in group)
        agreement = sum(record["agrees_with_manual"] for record in group)
        errors = sum(record["extraction_status"] != "ok" for record in group)
        low, high = wilson_interval(accepted, len(group))
        summary.append(
            {
                "variant": args.variant,
                "response_type": response_type,
                "n": len(group),
                "accepted": accepted,
                "acceptance_rate": accepted / len(group),
                "acceptance_wilson95_low": low,
                "acceptance_wilson95_high": high,
                "extraction_errors": errors,
                "agreement_with_manual": agreement,
                "agreement_rate": agreement / len(group),
            }
        )

    counts = Counter(record["response_type"] for record in records)
    by_task: dict[str, dict[str, int]] = defaultdict(dict)
    for record in records:
        by_task[record["ID"]][record["response_type"]] = record["accepted"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    prefix = f"extraction_{args.variant}_{stamp}"
    verdicts_path = args.output_dir / f"{prefix}_verdicts.csv"
    summary_path = args.output_dir / f"{prefix}_summary.csv"
    info_path = args.output_dir / f"{prefix}_run_info.json"
    write_csv(verdicts_path, records)
    write_csv(summary_path, summary)
    run_info = {
        "stage": "train_prompt_development",
        "final_test_run": False,
        "variant": args.variant,
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "model_file": args.model_file.name,
        "model_file_sha256": sha256_file(args.model_file),
        "quantization": "Q4_K_M",
        "data_file": args.data.name,
        "data_sha256": sha256_file(args.data),
        "n_verdicts": len(records),
        "response_type_counts": counts,
        "total_seconds": round(total_seconds, 3),
        "seconds_per_verdict": round(total_seconds / len(records), 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "decoding": {"temperature": 0, "top_p": 1, "max_tokens": 32, "seed": 42},
        "system_prompt": build_messages(rows[0], args.variant)[0]["content"],
        "few_shot_examples": (
            FEW_SHOT_EXAMPLES
            if args.variant == "few_shot"
            else BALANCED_CONTRASTIVE_EXAMPLES
            if args.variant == "contrastive_schema"
            else []
        ),
        "response_format": JSON_SCHEMA_RESPONSE_FORMAT if args.variant == "contrastive_schema" else None,
        "reference_visible_to_model": False,
        "comparison": "exact Decimal equality after final-answer extraction",
        "agreement_with_manual": sum(r["agrees_with_manual"] for r in records),
        "extraction_errors": sum(r["extraction_status"] != "ok" for r in records),
        "paired_control_attack_differences": sum(
            values.get("incorrect_control") != values.get("attack") for values in by_task.values()
        ),
        "output_files": {
            "verdicts": verdicts_path.as_posix(),
            "summary": summary_path.as_posix(),
        },
    }
    info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed {args.variant} in {total_seconds:.2f}s")
    print(verdicts_path)
    print(summary_path)
    print(info_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
