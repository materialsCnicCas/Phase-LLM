# -*- coding: utf-8 -*-
"""
Few-shot CoT Builder for Phase-LLM (Ablation Variant)

This script generates CoT data using a *single mixed prompt* with few-shot examples.
It is designed for ablation against role-decomposed multi-agent prompting.

Pipeline per sample:
1) Build a few-shot prompt from exemplar QA pairs + baseline mixed template.
2) Call teacher LLM and parse structured CoT JSON.
3) Verify predicted label against ground truth.
4) Run reflection strategies for limited rounds if prediction is wrong.
5) If still wrong, trigger label-guided fallback.
6) Reformat to natural reasoning and generate final response.

Security note:
- API credentials are NOT hardcoded. They are loaded by `call_llm.py` from env vars.
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


VALID_LABELS = {
    "Only FCC and L12 phases form",
    "Other phases form in addition to FCC and L12",
    "L12 phase does not form",
}


def read_text(path: Path) -> str:
    """Read UTF-8 text."""
    return path.read_text(encoding="utf-8")


def extract_json_block(text: str) -> Optional[str]:
    """Extract top-level JSON block from noisy model output."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def parse_cot_response(response: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Parse model response and validate `CoT` schema."""
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
    """Parse JSON with key `NaturalReasoning`."""
    try:
        candidate = response.strip()
        if not candidate.startswith("{"):
            candidate = extract_json_block(candidate) or ""
        obj = json.loads(candidate)
        natural_reasoning = obj["NaturalReasoning"]
        assert isinstance(natural_reasoning, str) and natural_reasoning.strip()
        return True, natural_reasoning
    except Exception:
        return False, None


def normalize_label(text: str) -> Optional[str]:
    """Map text to one of canonical labels if possible."""
    lowered = text.lower()
    if "only fcc and l12 phases form" in lowered:
        return "Only FCC and L12 phases form"
    if "other phases form in addition to fcc and l12" in lowered:
        return "Other phases form in addition to FCC and L12"
    if "l12 phase does not form" in lowered:
        return "L12 phase does not form"
    return None


def get_reasoning_stream(cot_steps: List[Dict[str, str]]) -> str:
    """Convert CoT list to markdown-like stream for reflection/reformat prompts."""
    rows: List[str] = []
    for step in cot_steps:
        title = step.get("title", step.get("action", "Step"))
        title = title.replace("Final Conclusion", "Conclusion")
        rows.append(f"### {title}\n{step.get('content', '')}")
    return "\n\n".join(rows).strip()


def extract_final_conclusion(cot_steps: List[Dict[str, str]]) -> str:
    """Extract final conclusion text from standard CoT shape."""
    return cot_steps[-2].get("content", "").strip()


def load_records(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load and normalize raw records from JSON/JSONL/CSV."""
    if dataset_path.suffix.lower() in {".json", ".jsonl", ".csv"} and dataset_path.exists():
        suffix = dataset_path.suffix.lower()
        kind = "csv" if suffix == ".csv" else "json"
        ds = load_dataset(kind, data_files={"train": str(dataset_path)})["train"]
    else:
        ds = None
        for suffix, kind in ((".json", "json"), (".jsonl", "json"), (".csv", "csv")):
            candidate = Path(str(dataset_path) + suffix)
            if candidate.exists():
                ds = load_dataset(kind, data_files={"train": str(candidate)})["train"]
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


def load_exemplars(exemplar_file: Path, max_k: int) -> List[Dict[str, str]]:
    """
    Load few-shot exemplars.

    Supported formats:
    - JSONL with keys: prompt/question + response/answer
    - JSON array with similar keys
    """
    if not exemplar_file.exists():
        raise FileNotFoundError(f"Few-shot exemplar file not found: {exemplar_file}")

    examples: List[Dict[str, str]] = []

    # Compatibility mode: original showcase folder with *.txt files.
    if exemplar_file.is_dir():
        for txt_file in sorted(exemplar_file.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            text = txt_file.read_text(encoding="utf-8")
            if "【Question】" not in text or "【Answer】" not in text:
                continue
            try:
                question_part = text.split("【Question】", 1)[1].split("【Answer】", 1)[0].strip()
                answer_part = text.split("【Answer】", 1)[1].strip()
                if question_part and answer_part:
                    examples.append({"question": question_part, "answer": answer_part})
            except Exception:
                continue

        if not examples:
            raise ValueError("No valid few-shot examples found in exemplar directory.")
        return examples[:max_k]

    if exemplar_file.suffix.lower() == ".jsonl":
        lines = exemplar_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            q = row.get("prompt", row.get("question", "")).strip()
            a = row.get("response", row.get("answer", "")).strip()
            if q and a:
                examples.append({"question": q, "answer": a})
    else:
        obj = json.loads(exemplar_file.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for row in obj:
                q = str(row.get("prompt", row.get("question", ""))).strip()
                a = str(row.get("response", row.get("answer", ""))).strip()
                if q and a:
                    examples.append({"question": q, "answer": a})

    if not examples:
        raise ValueError("No valid few-shot examples found.")

    return examples[:max_k]


def build_fewshot_prefix(exemplars: List[Dict[str, str]]) -> str:
    """Render few-shot examples into plain-text prefix."""
    blocks: List[str] = [
        "Below are solved reference examples. Learn the reasoning style and output format exactly."
    ]
    for i, item in enumerate(exemplars, start=1):
        blocks.append(
            f"\n[Example {i}]\n"
            f"Question:\n{item['question']}\n\n"
            f"Answer:\n{item['answer']}\n"
        )
    blocks.append("\nNow solve the next question following the same JSON schema.")
    return "\n".join(blocks)


class FewshotBuilder:
    """Few-shot CoT generation engine with reflection and fallback."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root_dir = Path(__file__).resolve().parent
        self.prompt_dir = self.root_dir / "prompts"
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Prompt templates
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
        self.exemplars = load_exemplars(Path(args.fewshot_file), args.num_shots)
        self.fewshot_prefix = build_fewshot_prefix(self.exemplars)

    def verify_prediction(self, generated_conclusion: str, label: str, trace: Dict[str, Any]) -> bool:
        """Use verifier prompt to check if generated answer matches ground truth."""
        query = self.prompt_verify.format(
            model_response=generated_conclusion,
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
        """Process one sample and save checkpoint file."""
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

            # Initial few-shot query: exemplar prefix + baseline mixed prompt.
            base_query = self.prompt_mixed.format(question=question)
            full_query = f"{self.fewshot_prefix}\n\n{base_query}"
            result["llm_queries"].append(full_query)

            ok, cot, raw = self.call_parse_cot(full_query)
            result["llm_responses"].append(raw)
            if not ok or cot is None:
                raise RuntimeError("Failed to get valid initial CoT from few-shot prompt.")

            result["Long_CoT"] = cot
            result["response_struct"].append(cot)
            result["response_type"].append("Init_Fewshot_Mixed")

            correct = self.verify_prediction(extract_final_conclusion(result["Long_CoT"]), label, result)

            # Reflection loop
            for reflection_idx in range(self.args.max_reflections):
                if correct:
                    break

                # First reflection excludes backtracking for diversity.
                if reflection_idx == 0 and len(self.strategy_pool) > 1:
                    strategy_name, strategy_prompt = random.choice(self.strategy_pool[1:])
                else:
                    strategy_name, strategy_prompt = random.choice(self.strategy_pool)

                reasoning = get_reasoning_stream(result["Long_CoT"][:-1])
                reflection_query = strategy_prompt.format(question=question, previous_reasoning=reasoning)

                result["llm_queries"].append(reflection_query)
                ok, recot, recot_raw = self.call_parse_cot(reflection_query)
                result["llm_responses"].append(recot_raw)
                if not ok or recot is None:
                    continue

                result["Long_CoT"] = result["Long_CoT"][:-1] + recot
                result["response_struct"].append(recot)
                result["response_type"].append(f"Reflection_{strategy_name}")

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

            # Final formatting
            if correct:
                stream = get_reasoning_stream(result["Long_CoT"])
                reformat_query = self.prompt_reformat.format(thought_process=stream, question=question)
                result["llm_queries"].append(reformat_query)

                reformat_raw = self.llm.retry_call(reformat_query)
                result["llm_responses"].append(reformat_raw)

                ok, natural_reasoning = parse_reformat_response(reformat_raw)
                if ok and natural_reasoning is not None:
                    result["Complex_CoT"] = natural_reasoning

                    final_query = self.prompt_final.format(internal_thinking=natural_reasoning, question=question)
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
    """Merge per-sample outputs and keep valid completion records only."""
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
    """Skip records that are already completed."""
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
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Build few-shot CoT data for ablation.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input dataset (.json/.jsonl/.csv or prefix).")
    parser.add_argument(
        "--fewshot_file",
        type=str,
        required=True,
        help="Path to few-shot exemplars (.json/.jsonl), typically from verified CoT samples.",
    )
    parser.add_argument("--num_shots", type=int, default=3, help="Number of exemplars to include in prefix.")
    parser.add_argument("--teacher_model", type=str, default=None, help="Optional teacher model override.")
    parser.add_argument("--max_reflections", type=int, default=5, help="Maximum reflection rounds.")
    parser.add_argument("--enable_label_fallback", action="store_true", help="Enable label-guided fallback.")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--limit_num", type=int, default=0, help="Debug cap; 0 means full dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic strategy sampling.")
    parser.add_argument("--output_dir", type=str, default="./output_data/fewshot", help="Per-sample checkpoint dir.")
    parser.add_argument(
        "--output_merged",
        type=str,
        default="./output_data/train_with_CoT_fewshot.json",
        help="Merged valid output JSON file.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_argparser().parse_args()
    random.seed(args.seed)

    records = load_records(Path(args.dataset))
    if args.limit_num > 0:
        records = records[: args.limit_num]

    print(f"Loaded records: {len(records)}")

    builder = FewshotBuilder(args)
    pending = deduplicate(records, builder.output_dir)

    print(f"Already completed: {len(records) - len(pending)}")
    print(f"Pending: {len(pending)}")

    if pending:
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            list(tqdm(pool.map(builder.process_one, pending), total=len(pending), desc="Building few-shot CoT"))

    merged = merge_valid_results(builder.output_dir)
    out_file = Path(args.output_merged)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Merged valid samples: {len(merged)}")
    print(f"Saved merged output: {out_file}")


if __name__ == "__main__":
    main()
