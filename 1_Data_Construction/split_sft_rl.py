# -*- coding: utf-8 -*-
"""
Split training JSONL into SFT and RL subsets with stratification.

Expected input schema per line:
- prompt: str
- response: str
- label: str (one of the three canonical labels)

Output:
- train_sft.jsonl
- train_rl.jsonl

Default split is 50/50 and stratified by label for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, str]]:
    """Read JSONL records."""
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(rows: List[Dict[str, str]], sft_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split rows into SFT/RL subsets while preserving label distribution."""
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    unlabeled: List[Dict[str, str]] = []

    for row in rows:
        label = str(row.get("label", "")).strip()
        if label:
            groups[label].append(row)
        else:
            unlabeled.append(row)

    rng = random.Random(seed)
    sft_rows: List[Dict[str, str]] = []
    rl_rows: List[Dict[str, str]] = []

    for label, items in groups.items():
        rng.shuffle(items)
        split_idx = int(len(items) * sft_ratio)
        # Ensure both subsets can receive samples when possible.
        if len(items) >= 2:
            split_idx = max(1, min(split_idx, len(items) - 1))

        sft_rows.extend(items[:split_idx])
        rl_rows.extend(items[split_idx:])

    # Unlabeled records are split evenly without stratification.
    if unlabeled:
        rng.shuffle(unlabeled)
        split_idx = int(len(unlabeled) * sft_ratio)
        sft_rows.extend(unlabeled[:split_idx])
        rl_rows.extend(unlabeled[split_idx:])

    rng.shuffle(sft_rows)
    rng.shuffle(rl_rows)
    return sft_rows, rl_rows


def label_stats(rows: List[Dict[str, str]]) -> Dict[str, int]:
    """Count labels for quick diagnostics."""
    stats: Dict[str, int] = defaultdict(int)
    for row in rows:
        stats[str(row.get("label", "")).strip()] += 1
    return dict(stats)


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    parser = argparse.ArgumentParser(description="Stratified split for SFT and RL training sets.")
    parser.add_argument("--input", type=str, default="C:\\Users\\admin\\Desktop\\构建reasoning\\Phase-LLM-Open-Source\\1_Data_Construction\\output_data\\train_multi_agent.jsonl", help="Input training JSONL path.")
    parser.add_argument("--sft_output", type=str, default="C:\\Users\\admin\\Desktop\\构建reasoning\\Phase-LLM-Open-Source\\1_Data_Construction\\output_data\\train_sft.jsonl", help="Output JSONL path for SFT subset.")
    parser.add_argument("--rl_output", type=str, default="C:\\Users\\admin\\Desktop\\构建reasoning\\Phase-LLM-Open-Source\\1_Data_Construction\\output_data\\train_rl.jsonl", help="Output JSONL path for RL subset.")
    parser.add_argument("--sft_ratio", type=float, default=0.5, help="Ratio allocated to SFT set (default: 0.5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    """Entrypoint."""
    args = parse_args()

    input_path = Path(args.input)
    sft_output = Path(args.sft_output)
    rl_output = Path(args.rl_output)

    rows = read_jsonl(input_path)
    sft_rows, rl_rows = stratified_split(rows, sft_ratio=args.sft_ratio, seed=args.seed)

    write_jsonl(sft_output, sft_rows)
    write_jsonl(rl_output, rl_rows)

    print(f"Input rows: {len(rows)}")
    print(f"SFT rows: {len(sft_rows)} -> {sft_output}")
    print(f"RL rows: {len(rl_rows)} -> {rl_output}")

    print("\nSFT label distribution:")
    for k, v in sorted(label_stats(sft_rows).items()):
        print(f"  {k or '<EMPTY_LABEL>'}: {v}")

    print("\nRL label distribution:")
    for k, v in sorted(label_stats(rl_rows).items()):
        print(f"  {k or '<EMPTY_LABEL>'}: {v}")


if __name__ == "__main__":
    main()
