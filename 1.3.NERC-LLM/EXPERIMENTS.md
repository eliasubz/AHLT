# Assignment 1.3 — LLM NERC Experiments

All commands are run from `1.3.NERC-LLM/bin/`.  
Results land in `1.3.NERC-LLM/results/`.  
Run `python3 analyze_results.py` at any point to see the current comparison table and plots.

---

## Phase 1 — Few-shot (ollama, local GPU or CPU)

```bash
cd 1.3.NERC-LLM/bin

# --- 1.1 Baseline: default prompt, random selection ---
sbatch fewshot.sh llama3.2:3b prompts01 5  train devel -ollama
sbatch fewshot.sh llama3.2:3b prompts01 10 train devel -ollama
sbatch fewshot.sh llama3.2:3b prompts01 15 train devel -ollama

# --- 1.2 Prompt variants (at fixed shot count) ---
sbatch fewshot.sh llama3.2:3b prompts02 10 train devel -ollama
sbatch fewshot.sh llama3.2:3b prompts03 10 train devel -ollama

# --- 1.3 Example selection strategy (diverse vs random) ---
sbatch fewshot.sh llama3.2:3b prompts01 10 train devel -ollama -diverse
sbatch fewshot.sh llama3.2:3b prompts03 10 train devel -ollama -diverse

# --- 1.4 Second model (pick best prompt from above) ---
sbatch fewshot.sh qwen2.5:7b  prompts01 10 train devel -ollama
sbatch fewshot.sh qwen2.5:7b  prompts03 10 train devel -ollama -diverse
```

---

## Phase 2 — Fine-tuning (Boada cluster, GPU required)

```bash
# --- 2.1 Train (saves weights to models/) ---
sbatch FT-train.sh llama3.2:3b prompts01 train devel
sbatch FT-train.sh llama3.2:3b prompts03 train devel

# --- 2.2 Inference on devel after training ---
sbatch FT-inference.sh llama3.2:3b prompts01 devel FT-llama3.2_3b-prompts01.weights
sbatch FT-inference.sh llama3.2:3b prompts03 devel FT-llama3.2_3b-prompts03.weights
```

---

## Analyze results (run after any phase)

```bash
python3 analyze_results.py
# prints ranked table + saves results/plots/*.png
```

---

## Final evaluation on test set (best config only — do this last!)

```bash
# Replace MODEL / PROMPTS / FLAGS with your best few-shot config
sbatch fewshot.sh llama3.2:3b prompts01 10 train test -ollama

# Replace weights dir with your best fine-tuned config
sbatch FT-inference.sh llama3.2:3b prompts01 test FT-llama3.2_3b-prompts01.weights

python3 analyze_results.py
```

---

## Expected outputs

| Phase | Experiments | New table rows | New plots |
|-------|-------------|----------------|-----------|
| 1.1 baseline (3 shot counts) | 3 | overview, per-type, shots-curve |
| 1.2 prompt variants | +2 | updated overview + per-type |
| 1.3 diverse selection | +2 | updated overview |
| 1.4 second model | +2 | updated overview |
| 2 fine-tuning | +2 | updated overview (FS vs FT visible) |
| **Total on devel** | **~11** | **3–5 PNG files** |
| + test set (best only) | +2 | separate overview_test.png |

---

## File naming convention

```
Few-shot:   FS-{model}-{prompt}-{shots}-{testdata}[-diverse][-quant].{out,stats,json}
Fine-tuning weights:  models/FT-{model}-{prompt}[-quant].weights/
Fine-tuning results:  FT-{model}-{prompt}[-quant]-{testdata}.{out,stats,json}
```

## Prompts summary

| File | Strategy |
|------|----------|
| `prompts01.json` | Baseline — detailed system prompt, multi-constraint user prompt |
| `prompts02.json` | Strict format — numbered rules, minimal user prompt, no extra output |
| `prompts03.json` | Drug_n focused — explicit drug vs drug_n decision rule with examples |
