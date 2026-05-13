# -*- coding: utf-8 -*-
"""
Zero-shot CoT Builder for Phase-LLM (Ablation Variant)

This script generates Chain-of-Thought (CoT) data using a zero-shot *mixed prompt*
(`prompts/baseline_mixed.txt`). It is used to compare against:
- role-decomposed multi-agent prompting, and
- few-shot mixed prompting.

Per-sample workflow:
1) Generate initial CoT from zero-shot mixed prompt.
2) Verify predicted label against ground-truth answer.
3) Run reflection strategies (up to max rounds) if prediction is wrong.
4) Apply label-guided fallback if still wrong.
5) Reformat CoT to natural reasoning and generate final response.

All API credentials are read via environment variables in `call_llm.py`.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from call_llm import TeacherLLM


def read_text(path: Path) -> str:
    """Read UTF-8 text from a file."""
    return path.read_text(encoding="utf-8")


def extract_json_block(text: str) -> Optional[str]:
    """Extract a JSON object block from free-form model output."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def parse_cot_response(response: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Parse a CoT JSON response and validate required tail actions.

    Required action order in the end:
    - Inner Thinking
    - Final Conclusion
    - Verification
    """
    try:
        candidate = response.strip()
        if not candidate.startswith("{"):
            candidate = extract_json_block(candidate) or ""
        obj = json.loads(candidate)

        cot = obj["CoT"]
        assert isinstance(cot, list) and len(cot) >= 3
        assert cot[-3]["action"] == "Inner Thinking"
        assert cot[-2]["action"] == "Final Conclusion"
        assert cot[-1]["action"] == "Verification"
        return True, obj
    except Exception:
        return False, None


def parse_reformat_response(response: str) -> Tuple[bool, Optional[str]]:
    """Parse reformat output JSON: {"NaturalReasoning": "..."}."""
    try:
        candidate = response.strip()
        if not candidate.startswith("{"):
            candidate = extract_json_block(candidate) or ""
        obj = json.loads(candidate)
        natural = obj["NaturalReasoning"]
        assert isinstance(natural, str) and natural.strip()
        return True, natural
    except Exception:
        return False, None


def normalize_label(text: str) -> Optional[str]:
    """Normalize answer text to canonical label if matched."""
    lowered = text.lower()
    if "only fcc and l12 phases form" in lowered:
        return "Only FCC and L12 phases form"
    if "other phases form in addition to fcc and l12" in lowered:
        return "Other phases form in addition to FCC and L12"
    if "l12 phase does not form" in lowered:
        return "L12 phase does not form"
    return None


def extract_final_conclusion(cot_steps: List[Dict[str, str]]) -> str:
    """Extract final conclusion from standard CoT list."""
    return cot_steps[-2].get("content", "").strip()


def get_reasoning_stream(cot_steps: List[Dict[str, str]]) -> str:
    """Convert structured CoT to markdown-like reasoning stream."""
    lines: List[str] = []
    for step in cot_steps:
        title = step.get("title", step.get("action", "Step"))
        title = title.replace("Final Conclusion", "Conclusion")
        lines.append(f"### {title}\n{step.get('content', '')}")
    return "\n\n".join(lines).strip()


def load_records(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON/JSONL/CSV and normalize schema.

    Output fields:
    - process_id
    - Open-ended Verifiable Question
    - Ground-True Answer
    """
    if dataset_path.suffix.lower() in {".json", ".jsonl", ".csv"} and dataset_path.exists():
        kind = "csv" if dataset_path.suffix.lower() == ".csv" else "json"
        ds = load_dataset(kind, data_files={"train": str(dataset_path)})["train"]
    else:
        ds = None
        for suffix, kind in ((".json", "json"), (".jsonl", "json"), (".csv", "csv")):
            p = Path(str(dataset_path) + suffix)
            if p.exists():
                ds = load_dataset(kind, data_files={"train": str(p)})["train"]
                break
        if ds is None:
            raise FileNotFoundError(f"Cannot find dataset {dataset_path}(.json/.jsonl/.csv)")

    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        question = row.get("Open-ended Verifiable Question", row.get("Question", ""))
        answer = row.get("Ground-True Answer", row.get("answer", row.get("label", "")))
        if not question or not answer:
            continue
        records.append(
            {
                "process_id": idx + 1,
                "Open-ended Verifiable Question": str(question).strip(),
                "Ground-True Answer": normalize_label(str(answer)) or str(answer).strip(),
            }
        )
    return records


class ZeroshotBuilder:
    """Zero-shot CoT generation engine with reflection and fallback."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root_dir = Path(__file__).resolve().parent
        self.prompt_dir = self.root_dir / "prompts"
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Prompts
        self.prompt_mixed = read_text(self.prompt_dir / "baseline_mixed.txt")
        self.prompt_verify = read_text(self.prompt_dir / "verify_prediction.txt")
        self.prompt_backtracking = read_text(self.prompt_dir / "supervisor_backtracking.txt")
        self.prompt_exploring = read_text(self.prompt_dir / "supervisor_exploring.txt")
        self.prompt_verification = read_text(self.prompt_dir / "supervisor_verification.txt")
        self.prompt_correction = read_text(self.prompt_dir / "supervisor_correction.txt")
        self.prompt_label_guided = read_text(self.prompt_dir / "supervisor_label_guided.txt")
        self.prompt_reformat = read_text(self.prompt_dir / "reformat_to_natural.txt")
        self.prompt_final = read_text(self.prompt_dir / "get_final_response.txt")

        self.strategy_pool: List[Tuple[str, str]] = [
            ("Backtracking", self.prompt_backtracking),
            ("Exploring", self.prompt_exploring),
            ("Verification", self.prompt_verification),
            ("Correction", self.prompt_correction),
        ]

        self.llm = TeacherLLM(model=args.teacher_model)

    def verify_prediction(self, generated: str, label: str, trace: Dict[str, Any]) -> bool:
        """Call verifier prompt and return boolean verdict."""
        query = self.prompt_verify.format(
            model_response=generated,
            reference_answer=label,
        )
        response = self.llm.retry_call(query)
        trace["llm_queries"].append(query)
        trace["llm_responses"].append(response)
        verdict = "true" in response.lower()
        trace["verify_history"].append(verdict)
        return verdict

    def call_parse_cot(self, query: str, retries: int = 2) -> Tuple[bool, Optional[List[Dict[str, str]]], str]:
        """Call LLM and parse CoT JSON with retry."""
        last = ""
        for _ in range(retries):
            last = self.llm.retry_call(query)
            ok, parsed = parse_cot_response(last)
            if ok and parsed is not None:
                return True, parsed["CoT"], last
        return False, None, last

    def process_one(self, sample: Dict[str, Any]) -> int:
        """Process one sample and write checkpoint JSON."""
        result = copy.deepcopy(sample)
        result.update(
            {
                "llm_queries": [],
                "llm_responses": [],
                "verify_history": [],
                "response_struct": [],
                "response_type": [],
                "Long_CoT": [],
                "Complex_CoT": "",
                "Response": "",
                "error": "",
            }
        )

        save_path = self.output_dir / f"{sample['process_id']}.json"

        try:
            question = sample["Open-ended Verifiable Question"]
            label = sample["Ground-True Answer"]

            # Initial zero-shot query (mixed baseline prompt).
            init_query = self.prompt_mixed.format(question=question)
            result["llm_queries"].append(init_query)

            ok, cot, raw = self.call_parse_cot(init_query)
            result["llm_responses"].append(raw)
            if not ok or cot is None:
                raise RuntimeError("Failed to get valid initial CoT in zero-shot mode.")

            result["Long_CoT"] = cot
            result["response_struct"].append(cot)
            result["response_type"].append("Init_Zeroshot_Mixed")

            correct = self.verify_prediction(extract_final_conclusion(result["Long_CoT"]), label, result)

            # Reflection loop
            for reflection_idx in range(self.args.max_reflections):
                if correct:
                    break

                if reflection_idx == 0 and len(self.strategy_pool) > 1:
                    name, strategy = random.choice(self.strategy_pool[1:])
                else:
                    name, strategy = random.choice(self.strategy_pool)

                reflection_query = strategy.format(
                    question=question,
                    previous_reasoning=get_reasoning_stream(result["Long_CoT"][:-1]),
                )
                result["llm_queries"].append(reflection_query)

                ok, recot, recot_raw = self.call_parse_cot(reflection_query)
                result["llm_responses"].append(recot_raw)
                if not ok or recot is None:
                    continue

                result["Long_CoT"] = result["Long_CoT"][:-1] + recot
                result["response_struct"].append(recot)
                result["response_type"].append(f"Reflection_{name}")

                correct = self.verify_prediction(extract_final_conclusion(result["Long_CoT"]), label, result)

            # Label-guided fallback
            if not correct and self.args.enable_label_fallback:
                fallback_query = self.prompt_label_guided.format(
                    question=question,
                    previous_reasoning=get_reasoning_stream(result["Long_CoT"][:-1]),
                    label=label,
                )
                result["llm_queries"].append(fallback_query)

                ok, fallback_cot, fallback_raw = self.call_parse_cot(fallback_query)
                result["llm_responses"].append(fallback_raw)
                if ok and fallback_cot is not None:
                    result["Long_CoT"] = result["Long_CoT"][:-1] + fallback_cot
                    result["response_struct"].append(fallback_cot)
                    result["response_type"].append("Label_Guided_Fallback")
                    result["verify_history"].append(True)
                    correct = True

            # Final reformat + final response
            if correct:
                stream = get_reasoning_stream(result["Long_CoT"])
                reformat_query = self.prompt_reformat.format(thought_process=stream, question=question)
                result["llm_queries"].append(reformat_query)

                reformat_raw = self.llm.retry_call(reformat_query)
                result["llm_responses"].append(reformat_raw)

                ok, natural = parse_reformat_response(reformat_raw)
                if ok and natural is not None:
                    result["Complex_CoT"] = natural

                    final_query = self.prompt_final.format(internal_thinking=natural, question=question)
                    result["llm_queries"].append(final_query)
                    final_resp = self.llm.retry_call(final_query)
                    result["llm_responses"].append(final_resp)
                    result["Response"] = final_resp

            result["Question"] = question
            result["GroundTruth"] = label

        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"

        save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1


def merge_valid_results(output_dir: Path) -> List[Dict[str, Any]]:
    """Merge per-sample JSON outputs, keeping only successful samples."""
    merged: List[Dict[str, Any]] = []
    for p in sorted(output_dir.glob("*.json"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem):
        try:
            item = json.loads(p.read_text(encoding="utf-8"))
            if item.get("Complex_CoT") and item.get("Response"):
                merged.append(item)
        except Exception:
            continue
    return merged


def deduplicate(records: List[Dict[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
    """Skip records that already have successful checkpoint files."""
    completed = set()
    for p in output_dir.glob("*.json"):
        try:
            item = json.loads(p.read_text(encoding="utf-8"))
            if item.get("Complex_CoT") and item.get("Response"):
                completed.add(int(item["process_id"]))
        except Exception:
            continue
    return [x for x in records if int(x["process_id"]) not in completed]


def build_argparser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="Build zero-shot CoT data for ablation.")
    parser.add_argument("--dataset", type=str, required=True, help="Input dataset path (.json/.jsonl/.csv or prefix).")
    parser.add_argument("--teacher_model", type=str, default=None, help="Optional teacher model override.")
    parser.add_argument("--max_reflections", type=int, default=5, help="Maximum reflection rounds.")
    parser.add_argument("--enable_label_fallback", action="store_true", help="Enable label-guided fallback.")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--limit_num", type=int, default=0, help="Debug cap; 0 means full dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="./output_data/zeroshot", help="Per-sample checkpoint dir.")
    parser.add_argument(
        "--output_merged",
        type=str,
        default="./output_data/train_with_CoT_zeroshot.json",
        help="Merged valid output JSON path.",
    )
    return parser


def main() -> None:
    """Program entrypoint."""
    args = build_argparser().parse_args()
    random.seed(args.seed)

    records = load_records(Path(args.dataset))
    if args.limit_num > 0:
        records = records[: args.limit_num]

    print(f"Loaded records: {len(records)}")

    builder = ZeroshotBuilder(args)
    pending = deduplicate(records, builder.output_dir)

    print(f"Already completed: {len(records) - len(pending)}")
    print(f"Pending: {len(pending)}")

    if pending:
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            list(tqdm(pool.map(builder.process_one, pending), total=len(pending), desc="Building zero-shot CoT"))

    merged = merge_valid_results(builder.output_dir)
    out_path = Path(args.output_merged)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Merged valid samples: {len(merged)}")
    print(f"Saved merged output: {out_path}")


if __name__ == "__main__":
    main()
