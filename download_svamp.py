r"""Export the original SVAMP subtraction splits and check the handwritten pilot.
"""

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
REVISION = "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"
SOURCE = f"https://huggingface.co/datasets/ChilleD/SVAMP/resolve/{REVISION}"
CACHE = ROOT / ".cache" / "svamp"
SOURCE_HASHES = {
    "train": "bb3b4cd2957f07643bbf9f8f8f5faba3bfb47fcb1d9034655b03060a76675e2b",
    "test": "9b5590fce66ea78ecfc7d0fc4a91fb801ffe289340269f8d9d28b21acad04ae6",
}
TASK_FIELDS = ["ID", "Body", "Question", "Equation", "Answer", "Type", "split"]
PILOT_FIELDS = ["case_id", *TASK_FIELDS, "response_type", "response_text",
                "expected_label", "manual_label"]
NUMBER = re.compile(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)")
SELECTED_IDS = (
    "chal-34", "chal-468", "chal-186", "chal-714", "chal-291",
    "chal-703", "chal-865", "chal-837", "chal-918", "chal-982",
)


def load_split(split):
    """Cache the two pinned JSON files; retain their original row order."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}.json"
    if not path.exists():
        with urlopen(f"{SOURCE}/{split}.json", timeout=60) as response:
            content = response.read()
        json.loads(content)  # Do not cache an invalid JSON download.
        path.write_bytes(content)
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == SOURCE_HASHES[split],
            f"Source checksum mismatch: {path}")
    return json.loads(content)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_pilot(tasks):
    """Check numeric invariants; prose and task meaning also need human review."""
    path = ROOT / "pilot_examples.csv"
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames == PILOT_FIELDS, "Unexpected pilot columns")
        rows = list(reader)
    require(len(rows) == 30, "Expected exactly 30 pilot rows")
    require(len({row['case_id'] for row in rows}) == 30, "Duplicate case_id")
    train = {task["ID"]: task for task in tasks if task["split"] == "train"}
    for index, task_id in enumerate(SELECTED_IDS):
        group = rows[index * 3:index * 3 + 3]
        task = train[task_id]
        answer = Decimal(str(task["Answer"]))
        # Selected equations are simple subtractions: avoid eval on source data.
        operands = re.fullmatch(r"\(\s*(\d+\.\d+)\s*-\s*(\d+\.\d+)\s*\)",
                                task["Equation"])
        require(operands is not None, f"Unexpected equation: {task_id}")
        left, right = map(Decimal, operands.groups())
        require(left - right == answer, f"Equation/Answer mismatch: {task_id}")
        response_types = ["correct_control", "incorrect_control", "attack"]
        for row, response_type in zip(group, response_types):
            require(row["ID"] == task_id, f"Wrong task order: {task_id}")
            for field in TASK_FIELDS:
                require(row[field] == str(task[field]), f"Changed {field}: {task_id}")
            require(row["response_type"] == response_type, "Wrong response order")
            require(row["manual_label"] in ("", "correct", "incorrect"),
                    "Unknown manual label")
            values = [Decimal(n) for n in NUMBER.findall(row["response_text"])]
            require(bool(values), f"Missing numbers: {row['case_id']}")
            # Each handwritten response explicitly gives its final answer last.
            expected = "correct" if values[-1] == answer else "incorrect"
            require(row["expected_label"] == expected, f"Wrong label: {row['case_id']}")
            require(expected == ("correct" if response_type == "correct_control"
                                 else "incorrect"), "Wrong control result")
        control, attack = [row["response_text"] for row in group[1:]]
        control_numbers = list(NUMBER.finditer(control))
        attack_numbers = list(NUMBER.finditer(attack))
        require(all(Decimal(m.group()) != answer for m in control_numbers),
                f"Reference number in incorrect control: {task_id}")
        require(len(control_numbers) == len(attack_numbers), "Changed number count")
        differences = [i for i, (a, b) in enumerate(zip(control_numbers, attack_numbers))
                       if a.group() != b.group()]
        require(len(differences) == 1, f"Expected one replaced number: {task_id}")
        changed = differences[0]
        require(changed < len(control_numbers) - 1, "Final answer changed")
        old, new = control_numbers[changed], attack_numbers[changed]
        require(Decimal(new.group()) == answer, "Replacement is not the reference")
        require(control[:old.start()] + new.group() + control[old.end():] == attack,
                f"Text changed outside the intermediate number: {task_id}")
        print(f"  {task_id}: intermediate {old.group()} -> {new.group()}, "
              f"wrong final {control_numbers[-1].group()}")
    print("Pilot OK: 10 train tasks, 30 responses, 10 single-number attack pairs.")


def main():
    tasks = []
    all_ids = set()
    for split, expected_total, expected_count in (("train", 700, 371), ("test", 300, 160)):
        rows = load_split(split)
        require(len(rows) == expected_total, f"Unexpected {split} size")
        for row in rows:
            require(row["ID"] not in all_ids, "Duplicate ID across source splits")
            all_ids.add(row["ID"])
        selected = [{**{field: row[field] for field in TASK_FIELDS if field != "split"},
                     "split": split} for row in rows if row["Type"] == "Subtraction"]
        require(len(selected) == expected_count, f"Unexpected {split} subtraction count")
        tasks.extend(selected)
        digest = hashlib.sha256((CACHE / f"{split}.json").read_bytes()).hexdigest()
        print(f"{split}: {len(rows)} total, {len(selected)} Subtraction; sha256={digest}")
    output = ROOT / "svamp_subtraction.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TASK_FIELDS)
        writer.writeheader()
        writer.writerows(tasks)
    print(f"Exported {len(tasks)} tasks to {output}")
    if (ROOT / "pilot_examples.csv").exists():
        check_pilot(tasks)


if __name__ == "__main__":
    main()
