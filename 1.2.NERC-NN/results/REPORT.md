# NERC Combined Report
**Advanced Human Language Technologies — Assignments 1.1, 1.2, 1.3**

> **How to use this file:**
> Run `python3 ../bin/analyze_results.py` after each experiment phase. It writes
> `results/summary_table.md` which you can paste into the sections below.
> Replace every `<!-- TODO -->` block with your actual numbers and observations.

---

## 1. Task Description

The task is **Named Entity Recognition and Classification (NERC)** of drug names in
biomedical text from the DDI Corpus (DrugBank + MedLine subsets).

Each entity mention must be identified (character offsets) and classified into one of
four types:

| Type | Description | Examples |
|------|-------------|---------|
| `drug` | Generic medicine approved for human use | metformin, ibuprofen |
| `group` | Family or class of medicines | antibiotics, NSAIDs |
| `brand` | Trade/proprietary name | Prozac, Tylenol |
| `drug_n` | Active substance not approved for human therapeutic use | heroin, cocaine |

**Evaluation metric:** Precision, Recall, F1 per entity type + macro F1.  
**Data split rule:** train → diagnose errors; devel → select configuration; test → final generalization only.

---

## 2. ML-based NERC (Assignment 1.1)

### 2.1 System Description

The ML approach uses handcrafted features fed to a linear classifier (CRF / SVM / MaxEnt)
with BIO tagging. Features include lexical, morphological, contextual, and external
dictionary (DrugBank, HSDB) information.

### 2.2 Experiments

<!-- TODO: paste your 1.1 results table here.
     Columns: classifier, feature_set, P, R, F1 on devel -->

| Classifier | Features | P | R | F1 |
|---|---|---|---|---|
| <!-- best --> | | | | |

### 2.3 Best Configuration

<!-- TODO: describe best model from 1.1 -->

- **Classifier:**
- **Key features:**
- **F1 on devel:**
- **F1 on test:**

---

## 3. NN-based NERC (Assignment 1.2)

### 3.1 System Description

A **bidirectional LSTM** with BIO tagging. Input is a concatenation of:
- Lowercased word embedding (`embLW`)
- Cased word embedding (`embW`)
- Suffix embedding (`embS`, suffix length = `suf_len`)
- Binary feature vector (orthographic + external dictionary features)

The output is a linear projection over LSTM hidden states → label distribution per token.
Training optimizes cross-entropy loss with Adam; best epoch selected by validation F1
(not accuracy, since ~98% tokens are `O`).

**Architecture (FlexibleNercLSTM):**
```
[embLW | embW | embS | features]  →  BiLSTM(hidden_size, num_layers)  →  Linear → Labels
```

### 3.2 Phase 1: Architecture Results

> Run: `python3 run.py train predict_all` then `python3 analyze_results.py`

<!-- TODO: paste contents of results/summary_table.md here after Phase 1 -->

| hidden | layers | dropout | emb | features | suf | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

**Observations:**
<!-- TODO: e.g. "Wider hidden size helps up to H=128, beyond that overfits.
     Single layer outperforms 2-layer on this dataset size." -->

**Best architecture:** `hidden=___  layers=___  dropout=___`

### 3.3 Phase 2: Feature Set Results

<!-- TODO: paste results/summary_table.md section after Phase 2 -->

| features | P | R | F1 | Δ F1 vs baseline |
|---|---|---|---|---|
| ortho+ext_full+ext_part (baseline) | | | | — |
| + drug_morph | | | | |
| + greek+length | | | | |
| + spacy | | | | |

**Observations:**
<!-- TODO: e.g. "Drug morphology features (+DM) give +2.1 F1.
     spaCy POS features added noise in this setup." -->

**Best feature set:** `___`

### 3.4 Phase 3: Suffix Length Results

<!-- TODO: paste relevant rows from results/summary_table.md after Phase 3 -->

| suf_len | P | R | F1 |
|---|---|---|---|
| 3 | | | |
| 5 | | | |
| 7 | | | |

**Best suffix length:** `___`

### 3.5 Per-type F1 Analysis

<!-- TODO: fill in per-type F1 for the best model -->

| Type | P | R | F1 |
|---|---|---|---|
| drug | | | |
| group | | | |
| brand | | | |
| drug_n | | | |
| **macro** | | | |

### 3.6 Error Analysis

<!-- TODO: inspect your best model's .out file against devel.xml and describe:
     - Most common error types (false positives / false negatives per class)
     - Is drug_n the hardest class? Why?
     - Do multi-word entities cause errors?
     - Are boundary detection errors common (I-tag mismatches)?
-->

Common error patterns observed:
1.
2.
3.

### 3.7 Best Configuration

- **Model:** `FlexibleNercLSTM`
- **hidden_size:**
- **num_layers:**
- **dropout:**
- **emb_sizes:**
- **features:**
- **suf_len:**
- **epochs / batch_size:**
- **F1 on devel:**
- **F1 on test:**

---

## 4. LLM-based NERC (Assignment 1.3)

### 4.1 System Description

Instead of BIO tagging, the LLM approach uses a **pseudo-XML representation**: the model
is asked to copy the input sentence and insert XML tags around drug mentions
(e.g. `<drug>aspirin</drug>`). This avoids the sub-word tokenization mismatch that makes
BIO unnatural for LLMs.

Two strategies:
- **Few-shot:** task description + `N` solved examples in the prompt; no weight update.
- **Fine-tuning:** LoRA adapters trained on all training examples.

### 4.2 Few-shot Results

> Run: `python3 analyze_results.py` from `1.3.NERC-LLM/bin/`

<!-- TODO: paste 1.3 results/summary.csv or summary_table.md here -->

| model | prompt | shots | selection | P | R | F1 |
|---|---|---|---|---|---|---|
| | | | | | | |

**Observations:**
<!-- TODO: e.g. "prompts03 (drug_n-focused) gave +1.8 F1.
     Diverse selection outperforms random at 10 shots.
     Performance plateaus after 15 shots due to context window." -->

### 4.3 Fine-tuning Results

<!-- TODO: paste FT rows from 1.3 results/summary.csv -->

| model | prompt | P | R | F1 |
|---|---|---|---|---|
| | | | | |

**Observations:**
<!-- TODO -->

### 4.4 Best Configuration

**Few-shot:**
- **Model:**
- **Prompt:**
- **Shots / selection:**
- **F1 on devel:**

**Fine-tuning:**
- **Base model:**
- **Prompt:**
- **LoRA config:**
- **F1 on devel:**

---

## 5. Comparison Across All Systems

> Best configuration per approach, evaluated on **devel** set.

<!-- TODO: fill in after all three assignments are complete -->

| System | Approach | P | R | F1 |
|---|---|---|---|---|
| 1.0 Baseline | Dictionary lookup | | | |
| 1.1 ML best | <!-- classifier --> | | | |
| 1.2 NN best | BiLSTM | | | |
| 1.3 LLM few-shot best | <!-- model/shots --> | | | |
| 1.3 LLM fine-tuned best | LoRA | | | |

> Best configuration per approach, evaluated on **test** set (final generalization).

| System | P | R | F1 |
|---|---|---|---|
| 1.1 ML best | | | |
| 1.2 NN best | | | |
| 1.3 LLM best | | | |

---

## 6. Conclusions

<!-- TODO: 3–5 sentences answering:
     - Which approach worked best overall and why?
     - What was the biggest single improvement in 1.2 (architecture vs features vs tuning)?
     - Did fine-tuning beat few-shot for 1.3? At what cost?
     - Which entity type was consistently hardest and why?
-->

Key findings:
1.
2.
3.
