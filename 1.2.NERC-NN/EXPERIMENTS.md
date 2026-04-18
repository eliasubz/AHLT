# Assignment 1.2 — NN NERC Experiments

All commands run from `1.2.NERC-NN/bin/`.  
Results land in `1.2.NERC-NN/results/`.  
Run `python3 analyze_results.py` at any point to see the current comparison table and plots.

---

## Step 0 — Parse XML (one-time, ~5–10 min due to spaCy)

```bash
cd 1.2.NERC-NN/bin
python3 run.py parse          # produces preprocessed/train.pck + devel.pck
python3 run.py parse test     # produces preprocessed/test.pck  (do last!)
```

---

## Phase 1 — Architecture sweep

`search_space` in `run.py` is already set for this phase.  
Varies: `hidden_size` × `num_layers` × `dropout` = **12 models**.

```bash
python3 run.py train epochs=10 batch_size=32
python3 run.py predict_all
python3 analyze_results.py
```

Models trained: `model_h{64|128|200}_l{1|2}_d{0.1|0.2}_e100-100-50_fOrEFEP_s5`

After running, look at `results/plots/arch_curve_devel.png` and `results/summary.csv`.  
Pick the best `hidden_size` / `num_layers` / `dropout` combination → call it **BEST_ARCH**.

---

## Phase 2 — Feature sweep

Open `run.py` and uncomment the **PHASE 2** block, filling in your BEST_ARCH values.  
Varies: 4 feature sets with best architecture = **4 models** (baseline already trained).

```bash
python3 run.py train epochs=10 batch_size=32
python3 run.py predict_all
python3 analyze_results.py
```

Feature sets tried:
| Short code | Groups active |
|---|---|
| `OrEFEP` | ortho + ext_full + ext_part (baseline) |
| `OrEFEPDM` | + drug_morph (pharmaceutical suffixes/prefixes) |
| `OrEFEPDMGkLn` | + greek letters + length |
| `OrEFEPDMGkLnSp` | + spaCy POS / stop-word flags |

Pick the best feature set → call it **BEST_FEATURES**.

---

## Phase 3 — Suffix length sweep

Open `run.py` and uncomment the **PHASE 3** block, filling in BEST_ARCH + BEST_FEATURES.  
Varies: `suf_len` ∈ {3, 5, 7} = **3 models** (best already trained).

```bash
python3 run.py train epochs=10 batch_size=32
python3 run.py predict_all
python3 analyze_results.py
```

---

## Final evaluation on test set (best config only — do this last!)

```bash
python3 run.py parse test    # if not done yet

# replace model name with your best config
python3 run.py predict test name=model_h128_l1_d0.2_e100-100-50_fOrEFEP_s5
python3 analyze_results.py
```

---

## Expected outputs

| Phase | Models | New rows | New plots |
|---|---|---|---|
| 1: Architecture | 12 | 12 | overview_devel, arch_curve_devel, per_type_devel |
| 2: Features | +3 | +3 | updated overview + per_type |
| 3: Suffix | +2 | +2 | updated overview |
| Final test | 1 | +1 | overview_test, per_type_test |
| **Total devel** | **~17** | **~17 rows** | **3 PNG files** |

---

## Model naming reference

```
model_h{H}_l{L}_d{D}_e{E}_f{F}_s{S}
        │    │    │    │    │    └─ suffix length
        │    │    │    │    └────── feature short codes (Or=ortho, EF=ext_full, ...)
        │    │    │    └─────────── embedding sizes (lc_word-word-suffix)
        │    │    └──────────────── dropout
        │    └───────────────────── num LSTM layers
        └────────────────────────── hidden size
```

## Feature short code reference

| Code | Full name | Description |
|---|---|---|
| `Or` | ortho | is_upper, is_title, is_digit, has_hyphen, has_number, has_punct |
| `EF` | ext_full | Full token match in DrugBank/HSDB (per type) |
| `EP` | ext_part | Partial match in DrugBank/HSDB multi-word entries |
| `DM` | drug_morph | Common pharmaceutical suffix/prefix patterns |
| `Gk` | greek | Contains a Greek letter (α, β, γ…) |
| `Ln` | length | Long word (>9 chars) / very short word (≤2 chars) |
| `Sp` | spacy | spaCy: is NOUN/PROPN, is_stop, is_alpha, like_num |
