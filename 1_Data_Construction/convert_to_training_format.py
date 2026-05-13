# -*- coding: utf-8 -*-
"""
Convert CoT construction outputs to training-ready JSONL format.

Input:
- A directory of per-sample JSON files (from build_cot_* scripts), OR
- A merged JSON file (list of sample dicts).

Output:
- JSONL with fields:
  {"prompt": <question>, "response": <complex_cot_or_final_response>, "label": <canonical_label>}

Why include `label`?
- SFT typically uses prompt/response only.
- RL reward construction and downstream evaluation need explicit labels.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CANONICAL_LABELS = {
    "Only FCC and L12 phases form",
    "Other phases form in addition to FCC and L12",
    "L12 phase does not form",
}


def normalize_label(text: str) -> Optional[str]:
    """Normalize free-form text into one of the three canonical labels."""
    if not text:
        return None
    lowered = text.lower()
    if "only fcc and l12 phases form" in lowered:
        return "Only FCC and L12 phases form"
    if "other phases form in addition to fcc and l12" in lowered:
        return "Other phases form in addition to FCC and L12"
    if "l12 phase does not form" in lowered:
        return "L12 phase does not form"
    return None


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def iter_samples(input_path: Path) -> Iterable[Dict[str, Any]]:
    """Yield sample dicts from merged file or per-sample directory."""
    if input_path.is_file():
        data = read_json(input_path)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            yield data
        return

    if input_path.is_dir():
        for file in sorted(input_path.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            try:
                item = read_json(file)
                if isinstance(item, dict):
                    yield item
            except Exception:
                continue


def pick_response(sample: Dict[str, Any], prefer_complex_cot: bool = True) -> Optional[str]:
    """
    Pick training response text.

    When `prefer_complex_cot=True`, concatenate Complex_CoT (thinking process)
    and Response (final answer) into a single training target:
        <think>...</think>\n\nFinal answer\n\nExplanation
    If only one field exists, fall back to using it alone.
    """
    complex_cot = str(sample.get("Complex_CoT", "")).strip()
    response = str(sample.get("Response", "")).strip()

    if prefer_complex_cot:
        if complex_cot and response:
            return complex_cot + "\n\n" + response
        return complex_cot or response or None
    else:
        if response and complex_cot:
            return complex_cot + "\n\n" + response
        return response or complex_cot or None


def pick_prompt(sample: Dict[str, Any]) -> Optional[str]:
    """Pick prompt/question text from possible fields."""
    prompt = (
        sample.get("Question")
        or sample.get("Open-ended Verifiable Question")
        or sample.get("prompt")
    )
    if not prompt:
        return None
    text = str(prompt).strip()
    return text if text else None


def pick_label(sample: Dict[str, Any], response_text: str) -> Optional[str]:
    """
    Pick and normalize label.

    Priority:
    1) GroundTruth / Ground-True Answer / label fields
    2) Extract from response text (first canonical phrase occurrence)
    """
    raw = (
        sample.get("GroundTruth")
        or sample.get("Ground-True Answer")
        or sample.get("label")
        or sample.get("answer")
    )
    label = normalize_label(str(raw)) if raw else None
    if label:
        return label

    return normalize_label(response_text)


def convert_samples(samples: Iterable[Dict[str, Any]], prefer_complex_cot: bool = True) -> List[Dict[str, str]]:
    """Convert raw samples to training JSONL rows."""
    rows: List[Dict[str, str]] = []
    for sample in samples:
        prompt = pick_prompt(sample)
        if not prompt:
            continue

        response = pick_response(sample, prefer_complex_cot=prefer_complex_cot)
        if not response:
            continue

        label = pick_label(sample, response)
        if label is None:
            # Keep rows without explicit label only if response is valid;
            # still useful for plain SFT. We fill with empty string to keep schema stable.
            label = ""

        rows.append({"prompt": prompt, "response": response, "label": label})

    return rows


def stratified_split(
    rows: List[Dict[str, str]],
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Stratified split by label: keep label distribution consistent in train/test.

    Returns (train_rows, test_rows).
    """
    rng = random.Random(seed)

    # Group by label
    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    for label, group in by_label.items():
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_ratio))  # at least 1 per label
        test_rows.extend(group[:n_test])
        train_rows.extend(group[n_test:])

    # Shuffle to avoid label-ordered blocks
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    return train_rows, test_rows


def save_jsonl(rows: List[Dict[str, str]], output_file: Path) -> None:
    """Save converted rows to JSONL."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    parser = argparse.ArgumentParser(description="Convert CoT outputs into training JSONL format.")
    parser.add_argument("--input", type=str,  help="Input merged JSON file or per-sample JSON directory.", default="C:\\Users\\admin\\Desktop\\构建reasoning\\Phase-LLM-Open-Source\\1_Data_Construction\\output_data\\multi_agent")
    parser.add_argument("--output", type=str,  help="Output JSONL path (train split).", default="C:\\Users\\admin\\Desktop\\构建reasoning\\Phase-LLM-Open-Source\\1_Data_Construction\\output_data\\train_multi_agent.jsonl")
    parser.add_argument(
        "--prefer_complex_cot",
        action="store_true",
        help="Prefer Complex_CoT over Response as training target (recommended for reasoning SFT).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of samples to hold out as test set (default: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint."""
    args = parse_args()
    input_path = Path(args.input)
    output_file = Path(args.output)

    # Step 1: Convert all samples
    all_rows = convert_samples(iter_samples(input_path), prefer_complex_cot=args.prefer_complex_cot)

    # Step 2: Stratified train/test split
    train_rows, test_rows = stratified_split(all_rows, test_ratio=args.test_ratio, seed=args.seed)

    # Step 3: Save train and test JSONL
    test_file = output_file.parent / output_file.name.replace("train_", "test_")
    if test_file == output_file:
        # Fallback: append _test before extension
        test_file = output_file.with_stem(output_file.stem + "_test")

    save_jsonl(train_rows, output_file)

    # Test set: only prompt + label (no reasoning/response needed)
    test_rows_clean = [{"prompt": r["prompt"], "label": r["label"]} for r in test_rows]
    save_jsonl(test_rows_clean, test_file)

    # Report
    train_labels = sum(1 for r in train_rows if r["label"] in CANONICAL_LABELS)
    test_labels = sum(1 for r in test_rows if r["label"] in CANONICAL_LABELS)

    print(f"Total converted: {len(all_rows)}")
    print(f"Train: {len(train_rows)} (with labels: {train_labels}) -> {output_file}")
    print(f"Test:  {len(test_rows)} (with labels: {test_labels}) -> {test_file}")

    # Label distribution
    from collections import Counter
    train_dist = Counter(r["label"] for r in train_rows)
    test_dist = Counter(r["label"] for r in test_rows)
    print(f"\nTrain label distribution: {dict(train_dist)}")
    print(f"Test  label distribution: {dict(test_dist)}")


if __name__ == "__main__":
    main()
