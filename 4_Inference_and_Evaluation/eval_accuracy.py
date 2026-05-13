# -*- coding: utf-8 -*-
"""
Evaluate prediction file and report Accuracy / Precision / Recall / Macro-F1.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


CANONICAL_LABELS = [
    "Only FCC and L12 phases form",
    "Other phases form in addition to FCC and L12",
    "L12 phase does not form",
]

LABEL_PATTERNS = {
    CANONICAL_LABELS[0]: [
        re.compile(r"only\s+fcc\s+and\s+l12\s+phases?\s+form", re.IGNORECASE),
        re.compile(r"only\s+fcc\s+and\s+l12\b", re.IGNORECASE),
    ],
    CANONICAL_LABELS[1]: [
        re.compile(r"other\s+phases?\s+form\s+in\s+addition\s+to\s+fcc\s+and\s+l12", re.IGNORECASE),
        re.compile(r"other\s+phases?.{0,40}fcc\s+and\s+l12", re.IGNORECASE | re.DOTALL),
    ],
    CANONICAL_LABELS[2]: [
        re.compile(r"l12\s+phase\s+does\s+not\s+form", re.IGNORECASE),
        re.compile(r"no\s+l12\s+phase", re.IGNORECASE),
    ],
}


def normalize_label(text: str) -> str:
    """Normalize free-form text into canonical labels when possible."""
    if not text:
        return ""
    text = str(text).strip()
    if text in CANONICAL_LABELS:
        return text

    best_pos = -1
    best_label = ""
    for label, patterns in LABEL_PATTERNS.items():
        for pattern in patterns:
            for m in pattern.finditer(text):
                if m.start() >= best_pos:
                    best_pos = m.start()
                    best_label = label

    return best_label or text


def read_prediction_jsonl(path: Path) -> List[Dict[str, str]]:
    """Read prediction rows from JSONL."""
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_metrics(rows: List[Dict[str, str]], normalize: bool = False) -> Dict[str, float]:
    """Compute standard classification metrics."""
    y_true = [str(r.get("ground_truth", "")).strip() for r in rows]
    y_pred = [str(r.get("prediction", "")).strip() for r in rows]

    if normalize:
        y_true = [normalize_label(x) for x in y_true]
        y_pred = [normalize_label(x) for x in y_pred]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "n_samples": len(rows),
    }


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate prediction JSONL.")
    parser.add_argument("--input", type=str, help="Prediction JSONL path.", default="/home/yuanyang/liangsihan/Phase-LLM-Open-Source/4_Inference_and_Evaluation/output_data/test_multi_agent_predictions.jsonl")
    parser.add_argument("--output", type=str, default="/home/yuanyang/liangsihan/Phase-LLM-Open-Source/4_Inference_and_Evaluation/output_data/test_multi_agent_metrics.json", help="Optional output metrics JSON path.")
    parser.add_argument("--normalize_labels", action=argparse.BooleanOptionalAction, default=True, help="Whether to normalize labels into canonical classes before scoring.")
    return parser.parse_args()


def main() -> None:
    """Entrypoint."""
    args = parse_args()

    rows = read_prediction_jsonl(Path(args.input))
    raw_metrics = compute_metrics(rows, normalize=False)
    norm_metrics = compute_metrics(rows, normalize=args.normalize_labels)

    metrics = {
        "raw": raw_metrics,
        "normalized": norm_metrics,
        "normalize_labels": bool(args.normalize_labels),
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
