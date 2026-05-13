# 0_Original_Data

This folder stores **original source data** used in the Phase-LLM pipeline, copied from the author workspace for reproducibility.

## raw/
Raw or near-raw tables before CoT construction and training conversion.

## processed/
Intermediate converted files that were used by the original scripts before final SFT/RL formatting.

## Canonical labels
- `L12 phase does not form`
- `Only FCC and L12 phases form`
- `Other phases form in addition to FCC and L12`

## Notes
- The Stage-1 scripts in `1_Data_Construction/` can consume CSV/JSON/JSONL and rebuild CoT outputs from these source files.
- If you need exact fold files used in a specific run, place them under `4_Inference_and_Evaluation/data_splits/`.
