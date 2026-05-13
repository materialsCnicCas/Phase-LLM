# -*- coding: utf-8 -*-
"""
Unified inference script for Phase-LLM.

Features:
- Load a local base model (Transformers) with optional LoRA adapter (PEFT).
- Run generation on JSON/JSONL/CSV datasets.
- Extract canonical phase label from model response.
- Save prediction records for downstream evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


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


def _extract_by_last_match(text: str) -> Optional[str]:
    """Find the last occurring label pattern in text and map it to canonical label."""
    if not text:
        return None

    best_pos = -1
    best_label = None
    for label, patterns in LABEL_PATTERNS.items():
        for pattern in patterns:
            for m in pattern.finditer(text):
                if m.start() >= best_pos:
                    best_pos = m.start()
                    best_label = label

    return best_label


def normalize_label(text: str) -> Optional[str]:
    """Normalize free-form text into one canonical label."""
    if not text:
        return None
    return _extract_by_last_match(text)


def extract_label_from_response(response: str) -> Optional[str]:
    """Extract canonical label from generated response."""
    if not response:
        return None

    # Prefer text after </think> tag if present.
    if "</think>" in response:
        after_think = response.split("</think>", 1)[1].strip()
        label = normalize_label(after_think)
        if label:
            return label

    # Prefer tail section first (many completions put final answer near the end).
    tail = response[-1500:]
    label = normalize_label(tail)
    if label:
        return label

    # Fallback: global search
    return normalize_label(response)


def _clean_question_for_heuristics(question: str) -> str:
    """Remove output-option block from question before heuristic fallback."""
    if not question:
        return ""

    split_markers = ["Output Format:", "Do not output any other text"]
    text = question
    for marker in split_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.lower()


def compress_repeated_response(response: str) -> str:
    """Compress repeated paragraphs to reduce looped generation noise."""
    if not response:
        return ""

    parts = [p.strip() for p in re.split(r"\n\s*\n", response) if p.strip()]
    if not parts:
        return response.strip()

    kept: List[str] = []
    for p in parts:
        if p not in kept:
            kept.append(p)
    return "\n\n".join(kept)


def force_classify_label(question: str, response: str) -> str:
    """Final fallback classifier to guarantee one canonical label."""
    response_text = (response or "").lower()
    question_text = _clean_question_for_heuristics(question or "")
    merged = f"{response_text}\n{question_text}"

    # Strong no-L12 cues
    no_l12_cues = [
        "l12 phase does not form",
        "no l12",
        "without aging",
        "no aging",
        "as-cast",
        "all temperatures are zero",
        "all temperatures and times are zero",
    ]
    if any(c in response_text for c in no_l12_cues):
        return CANONICAL_LABELS[2]

    # Strong additional-phase cues
    other_phase_keywords = [
        "other phases form in addition to fcc and l12",
        "in addition to fcc and l12",
        "sigma",
        "laves",
        "eta",
        "delta",
        "b2",
        "tcp",
    ]
    if any(k in response_text for k in other_phase_keywords):
        return CANONICAL_LABELS[1]

    # Strong only-FCC+L12 cues
    only_l12_cues = [
        "only fcc and l12 phases form",
        "only fcc and l12",
    ]
    if any(c in response_text for c in only_l12_cues):
        return CANONICAL_LABELS[0]

    # Aging-feature fallback from question text
    if any(k in merged for k in ["first aging temperature: 0", "first aging time: 0", "second aging temperature: 0", "second aging time: 0"]):
        return CANONICAL_LABELS[2]

    # Safe default: output the middle class when no clear signal exists.
    return CANONICAL_LABELS[0]


def extract_or_force_label(response: str, question: str) -> str:
    """Try standard extraction first, then force classification as final fallback."""
    compact = compress_repeated_response(response)
    return extract_label_from_response(compact) or force_classify_label(question, compact)


def load_records(path: Path) -> List[Dict[str, str]]:
    """Load evaluation records from JSON/JSONL/CSV."""
    suffix = path.suffix.lower()
    records: List[Dict[str, str]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                records.append(row)
        return records

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(x) for x in data]
        if isinstance(data, dict):
            return [dict(data)]

    if suffix == ".csv":
        import csv

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records

    raise ValueError(f"Unsupported file format: {path}")


class LocalPredictor:
    """Local model inference wrapper (vLLM backend)."""

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,  # 已将默认值修复为 1.0
        gpu_memory_utilization: float = 0.7,
        tensor_parallel_size: int = 1,
        max_lora_rank: int = 64,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.lora_request: Optional[LoRARequest] = None
        enable_lora = bool(lora_path)

        if lora_path:
            lora_dir = Path(lora_path).expanduser().resolve()
            if not lora_dir.exists():
                raise FileNotFoundError(f"LoRA path does not exist: {lora_dir}")
            self.lora_request = LoRARequest(
                lora_name=lora_dir.name,
                lora_int_id=1,
                lora_path=str(lora_dir),
            )
            warnings.warn(f"LoRA enabled: {lora_dir}", RuntimeWarning)

        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            enable_lora=enable_lora,
            max_lora_rank=max_lora_rank,
        )

        # 1. 必须先获取 tokenizer
        self.tokenizer = self.llm.get_tokenizer()
        
        # 2. 提取 Qwen / ChatML 格式的特殊终止符 ID
        stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
            
        for stop_str in ["<|im_end|>", "<|endoftext|>"]:
            tok_id = self.tokenizer.convert_tokens_to_ids(stop_str)
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                stop_token_ids.append(tok_id)

        # 3. 将 stop_token_ids 传入 SamplingParams
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            stop=["<|endoftext|>", "<|im_end|>"],
            stop_token_ids=list(set(stop_token_ids)),  # 解决因为模型不输出结束符造成的死循环
        )

    def _build_messages(self, question: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        return messages

    def _build_prompt(self, question: str, system_prompt: Optional[str] = None) -> str:
        messages = self._build_messages(question, system_prompt=system_prompt)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            if system_prompt:
                return f"System: {system_prompt}\n\nUser: {question}\nAssistant:"
            return question

    def generate(self, question: str, system_prompt: Optional[str] = None) -> str:
        """Generate one response for a question."""
        return self.generate_batch([question], system_prompt=system_prompt)[0]

    def generate_batch(self, questions: List[str], system_prompt: Optional[str] = None) -> List[str]:
        """Generate responses in batch via vLLM."""
        prompts = [self._build_prompt(q, system_prompt=system_prompt) for q in questions]
        outputs = self.llm.generate(prompts, self.sampling_params, lora_request=self.lora_request)
        return [out.outputs[0].text.strip() if out.outputs else "" for out in outputs]


def run_batch_inference(
    predictor: LocalPredictor,
    rows: List[Dict[str, str]],
    question_key: str,
    label_key: str,
    system_prompt: Optional[str],
) -> List[Dict[str, str]]:
    """Run inference on all rows and return prediction records."""
    outputs: List[Dict[str, str]] = []
    valid_questions: List[str] = []
    valid_rows: List[Dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        question = str(
            row.get(question_key)
            or row.get("Open-ended Verifiable Question")
            or row.get("prompt")
            or row.get("Question")
            or ""
        ).strip()

        if not question:
            continue

        valid_questions.append(question)
        valid_rows.append(row)

    raw_responses = predictor.generate_batch(valid_questions, system_prompt=system_prompt) if valid_questions else []

    for idx, (row, question, raw_response) in enumerate(zip(valid_rows, valid_questions, raw_responses), start=1):

        gt = str(row.get(label_key) or row.get("Ground-True Answer") or row.get("label") or "").strip()
        gt_norm = normalize_label(gt) or gt
        pred_label = extract_or_force_label(raw_response, question)

        outputs.append(
            {
                "id": str(row.get("id", idx)),
                "question": question,
                "ground_truth": gt_norm,
                "prediction": pred_label,
                "raw_response": raw_response,
            }
        )

    return outputs


def save_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    """Save rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    parser = argparse.ArgumentParser(description="Run inference for Phase-LLM models.")
    parser.add_argument("--model_path", type=str, help="Path to base/merged model.", default="/home/yuanyang/liangsihan/Phase-LLM-Open-Source/2_Training_SFT/saves/phase-llm-qwen3-8b-sft-merged")
    parser.add_argument("--lora_path", type=str, default="/home/yuanyang/liangsihan/Phase-LLM-Open-Source/3_Training_RL/output/phase_llm_grpo/test/checkpoint-440", help="Optional LoRA adapter path for vLLM LoRA inference.")
    parser.add_argument("--input", type=str, help="Input JSON/JSONL/CSV dataset path.", default="/home/yuanyang/liangsihan/Phase-LLM-Open-Source/1_Data_Construction/output_data/test_process.jsonl")
    parser.add_argument("--output", type=str, help="Output prediction JSONL path.", default="/home/yuanyang/liangsihan/Phase-LLM-Open-Source/4_Inference_and_Evaluation/output_data/test_multi_agent_predictions.jsonl")
    parser.add_argument("--question_key", type=str, default="prompt", help="Key name for question text.")
    parser.add_argument("--label_key", type=str, default="label", help="Key name for label text.")
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.00) # 将这里默认值改回了 1.00
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_lora_rank", type=int, default=32)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a senior metallurgist specializing in phase prediction for complex alloys. First, please proceed with your reasoning step-by-step within `<think>...</think>`. Then, on a new line, provide a final label that can only be one of the following: Only the BCC phase is formed; Other phases besides the BCC phase are formed. Do not add new labels or rewrite the final label.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint."""
    args = parse_args()

    predictor = LocalPredictor(
        model_path=args.model_path,
        lora_path=args.lora_path or None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_lora_rank=args.max_lora_rank,
    )

    rows = load_records(Path(args.input))
    outputs = run_batch_inference(
        predictor=predictor,
        rows=rows,
        question_key=args.question_key,
        label_key=args.label_key,
        system_prompt=args.system_prompt,
    )

    save_jsonl(Path(args.output), outputs)
    think_count = sum(1 for row in outputs if "<think>" in str(row.get("raw_response", "")))
    think_rate = think_count / len(outputs) if outputs else 0.0
    print(f"Input samples: {len(rows)}")
    print(f"Predictions saved: {len(outputs)}")
    print(f"Responses with <think>: {think_count}/{len(outputs)} ({think_rate:.1%})")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
