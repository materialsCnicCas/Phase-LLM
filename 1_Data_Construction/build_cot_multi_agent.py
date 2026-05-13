# -*- coding: utf-8 -*-
"""
Multi-Agent CoT Builder for Phase-LLM (Core Script)

This script builds high-quality Chain-of-Thought (CoT) data with a robust pipeline:
1) Call teacher LLM with thermodynamic agent prompt.
2) Call teacher LLM with kinetics agent prompt (conditioned on thermodynamic analysis).
3) Verify predicted label against ground truth.
4) If incorrect, run multi-round reflection (up to configurable max, default=5).
5) If still incorrect, trigger label-guided fallback to force scientifically grounded correction.
6) Reformat reasoning and generate final response for SFT/RL training.

Design goals:
- Reproducibility: deterministic seed + explicit IO schema.
- Robustness: strict JSON parsing + retries + per-sample checkpointing.
- Transparency: store all prompts/responses and verification traces.

Environment variables required by `call_llm.py`:
- TEACHER_API_KEY
- TEACHER_BASE_URL (optional; default is DashScope compatible endpoint)
- TEACHER_MODEL (optional; default is qwen-plus)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from call_llm import TeacherLLM


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
VALID_LABELS = {
    "Only FCC and L12 phases form",
    "Other phases form in addition to FCC and L12",
    "L12 phase does not form",
}

PROMPT_FILES = {
    "agent_thermo": "agent_thermo.txt",
    "agent_kinetic": "agent_kinetic.txt",
    "agent_synthesis": "agent_synthesis.txt",
    "verify": "verify_prediction.txt",
    "backtracking": "supervisor_backtracking.txt",
    "exploring": "supervisor_exploring.txt",
    "verification": "supervisor_verification.txt",
    "correction": "supervisor_correction.txt",
    "label_guided": "supervisor_label_guided.txt",
    "reformat": "reformat_to_natural.txt",
    "final_response": "get_final_response.txt",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def read_text(path: Path) -> str:
    """Read UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def extract_json_block(text: str) -> Optional[str]:
    """
    Extract the first top-level JSON object from model output.

    Many chat models return extra prose around JSON. This function attempts to
    recover the JSON object by locating the first '{' and the last '}'.
    """
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def parse_cot_response(response: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Parse JSON response and validate core CoT structure.

    Required shape:
    {
      "CoT": [
        ..., 
        {"action": "Inner Thinking", ...},
        {"action": "Final Conclusion", "content": ...},
        {"action": "Verification", "content": ...}
      ]
    }
    """
    try:
        candidate = response.strip()
        if not candidate.startswith("{"):
            candidate = extract_json_block(candidate) or ""
        obj = json.loads(candidate)

        cot = obj["CoT"]
        assert isinstance(cot, list) and len(cot) >= 3, "CoT must be a non-empty list with >= 3 steps"
        assert cot[-3]["action"] == "Inner Thinking", "Third-last action must be Inner Thinking"
        assert cot[-2]["action"] == "Final Conclusion", "Second-last action must be Final Conclusion"
        assert cot[-1]["action"] == "Verification", "Last action must be Verification"
        return True, obj
    except Exception:
        return False, None


def parse_reformat_response(response: str) -> Tuple[bool, Optional[str]]:
    """Parse reformat JSON output: {"NaturalReasoning": "..."}."""
    try:
        candidate = response.strip()
        if not candidate.startswith("{"):
            candidate = extract_json_block(candidate) or ""
        obj = json.loads(candidate)
        natural = obj["NaturalReasoning"]
        assert isinstance(natural, str) and len(natural.strip()) > 0
        return True, natural
    except Exception:
        return False, None


def get_reasoning_stream(cot_steps: List[Dict[str, str]]) -> str:
    """
    Convert structured CoT list to readable markdown-like stream.
    This stream is fed into reflection and reformat prompts.
    """
    lines: List[str] = []
    for step in cot_steps:
        title = step.get("title", step.get("action", "Step"))
        if title == "Final Conclusion":
            title = "Conclusion"
        content = step.get("content", "")
        lines.append(f"### {title}\n{content}")
    return "\n\n".join(lines).strip()


def extract_final_conclusion(cot_steps: List[Dict[str, str]]) -> str:
    """Extract conclusion text from the second-last CoT item."""
    return cot_steps[-2].get("content", "").strip()


def normalize_label(text: str) -> Optional[str]:
    """
    Map arbitrary text to one of three canonical labels if possible.
    Returns None if no label can be recognized.
    """
    lowered = text.lower()

    if "only fcc and l12 phases form" in lowered:
        return "Only FCC and L12 phases form"
    if "other phases form in addition to fcc and l12" in lowered:
        return "Other phases form in addition to FCC and L12"
    if "l12 phase does not form" in lowered:
        return "L12 phase does not form"

    return None


def load_dataset_records(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON/JSONL/CSV and normalize to canonical schema.

    Canonical fields per record:
    - process_id
    - Open-ended Verifiable Question
    - Ground-True Answer
    - Instruction
    """
    records: List[Dict[str, Any]] = []

    # Prefer explicit suffix if provided.
    if dataset_path.suffix.lower() in {".json", ".jsonl", ".csv"} and dataset_path.exists():
        suffix = dataset_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            ds = load_dataset("json", data_files={"train": str(dataset_path)})["train"]
        else:
            ds = load_dataset("csv", data_files={"train": str(dataset_path)})["train"]
    else:
        # Fallback: try appending common suffixes.
        for suffix, kind in ((".json", "json"), (".jsonl", "json"), (".csv", "csv")):
            candidate = Path(str(dataset_path) + suffix)
            if candidate.exists():
                ds = load_dataset(kind, data_files={"train": str(candidate)})["train"]
                break
        else:
            raise FileNotFoundError(
                f"Cannot find dataset file: {dataset_path}(.json/.jsonl/.csv)"
            )

    for idx, row in enumerate(ds):
        question = row.get("Open-ended Verifiable Question", row.get("Question", ""))
        answer = row.get("Ground-True Answer", row.get("answer", row.get("label", "")))
        instruction = row.get("Instruction", "No")

        if not question or not answer:
            continue

        normalized_answer = normalize_label(str(answer).strip()) or str(answer).strip()

        records.append(
            {
                "process_id": idx + 1,
                "Open-ended Verifiable Question": str(question).strip(),
                "Ground-True Answer": normalized_answer,
                "Instruction": str(instruction).strip() if instruction is not None else "No",
            }
        )

    return records


# -----------------------------------------------------------------------------
# Core builder
# -----------------------------------------------------------------------------
class MultiAgentCoTBuilder:
    """Encapsulates full CoT data construction workflow for one dataset."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root_dir = Path(__file__).resolve().parent
        self.prompt_dir = self.root_dir / "prompts"
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load prompts once for efficiency and reproducibility.
        self.prompts = {name: read_text(self.prompt_dir / file_name) for name, file_name in PROMPT_FILES.items()}

        # Reflection strategy pool.
        self.search_strategies: List[Tuple[str, str]] = [
            ("Backtracking", self.prompts["backtracking"]),
            ("Exploring", self.prompts["exploring"]),
            ("Verification", self.prompts["verification"]),
            ("Correction", self.prompts["correction"]),
        ]

        # Teacher model client.
        self.llm = TeacherLLM(
            model=args.teacher_model,
            api_key=args.api_key,
            base_url=args.base_url,
            port=args.port,
        )

    def verify_prediction(self, final_conclusion: str, ground_truth: str, trace: Dict[str, Any]) -> bool:
        """
        Verify whether generated conclusion matches reference label using teacher LLM.

        Returns True if verifier outputs "True"; otherwise False.
        """
        prompt = self.prompts["verify"].format(
            model_response=final_conclusion,
            reference_answer=ground_truth,
        )
        response = self.llm.retry_call(prompt)
        trace["llm_queries"].append(prompt)
        trace["llm_responses"].append(response)

        verdict = "true" in response.lower()
        trace["verify_history"].append(verdict)
        return verdict

    def call_and_parse_cot(self, prompt: str, max_retries: int = 2) -> Tuple[bool, Optional[List[Dict[str, str]]], str]:
        """Call model and parse JSON CoT with bounded retries."""
        last_response = ""
        for _ in range(max_retries):
            last_response = self.llm.retry_call(prompt)
            ok, parsed = parse_cot_response(last_response)
            if ok and parsed is not None:
                return True, parsed["CoT"], last_response
        return False, None, last_response

    def process_one(self, sample: Dict[str, Any]) -> int:
        """
        Process one sample and save per-sample JSON checkpoint.

        Return 1 for success path completion, even if sample fails internally;
        errors are recorded in output JSON for post-mortem analysis.
        """
        save_path = self.output_dir / f"{sample['process_id']}.json"

        # Skip if a successful checkpoint already exists (do not overwrite)
        if save_path.exists():
            try:
                existing = json.loads(save_path.read_text(encoding="utf-8"))
                if existing.get("Complex_CoT") and existing.get("Response"):
                    return 1  # Already completed successfully
            except Exception:
                pass  # Corrupted file, re-process

        result = copy.deepcopy(sample)
        result.update(
            {
                "llm_queries": [],
                "llm_responses": [],
                "verify_history": [],
                "response_type": [],
                "response_struct": [],
                "Long_CoT": [],
                "Complex_CoT": "",
                "Response": "",
                "error": "",
            }
        )

        try:
            question = sample["Open-ended Verifiable Question"]
            label = sample["Ground-True Answer"]

            # -----------------------------------------------------------------
            # Step A: Sequential agent execution (Thermo -> Kinetic -> Synthesis)
            # -----------------------------------------------------------------
            thermo_prompt = self.prompts["agent_thermo"].format(question=question)
            result["llm_queries"].append(thermo_prompt)
            ok, thermo_cot, thermo_raw = self.call_and_parse_cot(thermo_prompt)
            result["llm_responses"].append(thermo_raw)
            if not ok or thermo_cot is None:
                raise RuntimeError("Failed to generate valid thermodynamic CoT JSON.")

            result["response_struct"].append(thermo_cot)
            result["response_type"].append("Init_Thermodynamic_Agent")

            thermo_summary = get_reasoning_stream(thermo_cot)
            kinetic_instruction = (
                "Use the previous thermodynamic analysis as context, then refine kinetics and processing arguments.\n\n"
                f"Previous reasoning:\n{thermo_summary}"
            )

            kinetic_prompt = self.prompts["agent_kinetic"].format(
                question=question,
                instruction=kinetic_instruction,
            )
            result["llm_queries"].append(kinetic_prompt)
            ok, kinetic_cot, kinetic_raw = self.call_and_parse_cot(kinetic_prompt)
            result["llm_responses"].append(kinetic_raw)
            if not ok or kinetic_cot is None:
                raise RuntimeError("Failed to generate valid kinetic CoT JSON.")

            result["response_struct"].append(kinetic_cot)
            result["response_type"].append("Init_Kinetics_Agent")

            # Optional synthesis re-pass to stabilize final structure.
            synthesis_instruction = (
                "Integrate thermodynamic + kinetics evidence into one coherent scientific judgment.\n\n"
                f"Thermodynamic stream:\n{thermo_summary}\n\n"
                f"Kinetic stream:\n{get_reasoning_stream(kinetic_cot)}"
            )
            synthesis_prompt = self.prompts["agent_synthesis"].format(
                question=question,
                instruction=synthesis_instruction,
            )
            result["llm_queries"].append(synthesis_prompt)
            ok, synthesis_cot, synthesis_raw = self.call_and_parse_cot(synthesis_prompt)
            result["llm_responses"].append(synthesis_raw)
            if ok and synthesis_cot is not None:
                current_cot = synthesis_cot
                result["response_struct"].append(synthesis_cot)
                result["response_type"].append("Init_Synthesis_Agent")
            else:
                # Fallback to kinetic output if synthesis stage is malformed.
                current_cot = kinetic_cot

            result["Long_CoT"] = current_cot

            # -----------------------------------------------------------------
            # Step B: Verification + multi-round reflection
            # max_reflections controls the TOTAL number of verifications
            # (including the initial one), not just the reflection rounds.
            # -----------------------------------------------------------------
            is_correct = self.verify_prediction(extract_final_conclusion(current_cot), label, result)
            total_verifications = 1  # count the initial verification above

            while not is_correct and total_verifications < self.args.max_reflections:
                reflection_round = total_verifications  # 1-based reflection index

                # First reflection excludes Backtracking to encourage diversification,
                # mirroring the convention used in your original scripts.
                if reflection_round == 1 and len(self.search_strategies) > 1:
                    strategy_name, strategy_prompt = random.choice(self.search_strategies[1:])
                else:
                    strategy_name, strategy_prompt = random.choice(self.search_strategies)

                reasoning_stream = get_reasoning_stream(result["Long_CoT"][:-1])
                reflection_query = strategy_prompt.format(
                    question=question,
                    previous_reasoning=reasoning_stream,
                )

                result["llm_queries"].append(reflection_query)
                ok, reflection_cot, reflection_raw = self.call_and_parse_cot(reflection_query)
                result["llm_responses"].append(reflection_raw)
                if not ok or reflection_cot is None:
                    total_verifications += 1
                    continue

                # Merge strategy continuation into previous reasoning trace.
                result["Long_CoT"] = result["Long_CoT"][:-1] + reflection_cot
                result["response_struct"].append(reflection_cot)
                result["response_type"].append(f"Reflection_{strategy_name}")

                is_correct = self.verify_prediction(
                    extract_final_conclusion(result["Long_CoT"]),
                    label,
                    result,
                )
                total_verifications += 1

            # -----------------------------------------------------------------
            # Step C: Label-guided fallback (guaranteed rescue path)
            # -----------------------------------------------------------------
            if not is_correct and self.args.enable_label_fallback:
                fallback_query = self.prompts["label_guided"].format(
                    question=question,
                    previous_reasoning=get_reasoning_stream(result["Long_CoT"][:-1]),
                    label=label,
                )

                result["llm_queries"].append(fallback_query)
                ok, fallback_cot, fallback_raw = self.call_and_parse_cot(fallback_query)
                result["llm_responses"].append(fallback_raw)
                if ok and fallback_cot is not None:
                    result["Long_CoT"] = result["Long_CoT"][:-1] + fallback_cot
                    result["response_struct"].append(fallback_cot)
                    result["response_type"].append("Label_Guided_Fallback")
                    # Treat fallback as accepted terminal state.
                    result["verify_history"].append(True)
                    is_correct = True

            # -----------------------------------------------------------------
            # Step D: Reformat and finalize response
            # -----------------------------------------------------------------
            if is_correct:
                stream = get_reasoning_stream(result["Long_CoT"])
                reformat_query = self.prompts["reformat"].format(
                    thought_process=stream,
                    question=question,
                )

                result["llm_queries"].append(reformat_query)
                reformat_raw = self.llm.retry_call(reformat_query)
                result["llm_responses"].append(reformat_raw)

                ok, natural_reasoning = parse_reformat_response(reformat_raw)
                if ok and natural_reasoning is not None:
                    result["Complex_CoT"] = natural_reasoning

                    final_query = self.prompts["final_response"].format(
                        internal_thinking=natural_reasoning,
                        question=question,
                    )
                    result["llm_queries"].append(final_query)
                    final_response = self.llm.retry_call(final_query)
                    result["llm_responses"].append(final_response)
                    result["Response"] = final_response

            result["Question"] = question
            result["GroundTruth"] = label

        except Exception as exc:  # noqa: BLE001 (intentional broad catch for batch robustness)
            result["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        # Always save checkpoint, even on failure.
        save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1


def merge_valid_results(output_dir: Path) -> List[Dict[str, Any]]:
    """Merge per-sample JSON files and keep only successful samples."""
    merged: List[Dict[str, Any]] = []
    for file in sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        try:
            item = json.loads(file.read_text(encoding="utf-8"))
            if item.get("Complex_CoT") and item.get("Response"):
                merged.append(item)
        except Exception:
            continue
    return merged


def deduplicate_by_process_id(records: List[Dict[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
    """Skip records that already have a successful checkpoint file."""
    done_ids = set()
    for file in output_dir.glob("*.json"):
        try:
            item = json.loads(file.read_text(encoding="utf-8"))
            if item.get("Complex_CoT") and item.get("Response"):
                done_ids.add(int(item.get("process_id")))
        except Exception:
            continue
    return [r for r in records if int(r["process_id"]) not in done_ids]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(description="Build multi-agent CoT data for Phase-LLM.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="C:\\Users\\admin\\Desktop\\构建reasoning\\Phase-LLM-Open-Source\\0_Original_Data\\processed\\train_dual_task_L12_converted.csv",
        help="Path to input dataset file (.json/.jsonl/.csv) or path prefix without suffix.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".\\Phase-LLM-Open-Source\\1_Data_Construction\\output_data\\multi_agent",
        help="Directory to store per-sample JSON checkpoints.",
    )
    parser.add_argument(
        "--output_merged",
        type=str,
        default="./output_data/train_with_CoT_multi_agent.json",
        help="Path for merged successful output JSON.",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default=os.environ.get("TEACHER_MODEL", "deepseek-reasoner"),
        help="Optional override for TEACHER_MODEL environment variable.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-",
        help="Optional API key. If omitted, keys are read from environment variables.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.deepseek.com",
        help="Optional API base URL (without port is fine). If omitted, env/default is used.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="",
        help="Optional API port, independent from base_url (e.g., 4780, 8000).",
    )
    parser.add_argument(
        "--max_reflections",
        type=int,
        default=4,
        help="Maximum reflection rounds before fallback.",
    )
    parser.add_argument(
        "--enable_label_fallback",
        action="store_true",
        default=True,
        help="Enable label-guided fallback after max reflections (enabled by default).",
    )
    parser.add_argument(
        "--disable_label_fallback",
        action="store_false",
        dest="enable_label_fallback",
        help="Disable label-guided fallback after max reflections.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Parallel worker count for sample processing.",
    )
    parser.add_argument(
        "--limit_num",
        type=int,
        default=0,
        help="Optional cap for debugging; 0 means process all records.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for strategy sampling.",
    )
    return parser


def main() -> None:
    """Program entrypoint."""
    parser = build_argparser()
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_path = Path(args.dataset)
    records = load_dataset_records(dataset_path)

    if args.limit_num > 0:
        records = records[: args.limit_num]

    print(f"Loaded records: {len(records)}")

    builder = MultiAgentCoTBuilder(args)
    pending = deduplicate_by_process_id(records, builder.output_dir)

    print(f"Already completed: {len(records) - len(pending)}")
    print(f"Pending: {len(pending)}")

    if pending:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(builder.process_one, pending), total=len(pending), desc="Building CoT"))

    merged = merge_valid_results(builder.output_dir)
    output_merged = Path(args.output_merged)
    output_merged.parent.mkdir(parents=True, exist_ok=True)
    output_merged.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Merged valid samples: {len(merged)}")
    print(f"Saved merged file: {output_merged}")


if __name__ == "__main__":
    main()
