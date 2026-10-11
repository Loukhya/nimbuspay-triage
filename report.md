# Technical Report: Ticket Triage LLM Fine-Tuning

**Candidate Assessment:** ML Intern – LLM Fine-Tuning (NimbusPay Ticket Triage)  
**Model Architecture:** `Qwen/Qwen2.5-1.5B-Instruct` + 4-bit QLoRA  
**Best Result on Dev:** **58.5% Exact Match**, **92.2% Mean Field Accuracy**  

---

## 1. Executive Summary
NimbusPay requires an automated ticket triage system that maps free-text customer queries into an 8-key JSON schema with strict business logic (priority escalation, date resolution, currency conversion into paise, transaction ID normalization). Using an open-weights model constrained to $\le$ 4B parameters and $\le$ 50M trainable parameters on a single Google Colab T4 GPU, we fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` using QLoRA. 

A thorough data audit of `train.jsonl` uncovered 209 critical anomalies (syntax errors, duplicate records, key misnomers, and rule violations). Cleaning this dataset yielded a direct **+3.0% absolute improvement in Exact Match** over the raw data baseline (55.5% $\rightarrow$ 58.5%), reaching **92.2% Mean Field Accuracy**.

---

## 2. Part A: Data Audit & Cleaning Findings
The provided `data/train.jsonl` (1,445 rows) reflected realistic pipeline noise. Running custom schema verification scripts revealed multiple classes of errors recorded in `cleaning_log.csv`:

| Issue Identified | Count | Action | Justification |
|---|:---:|:---:|---|
| **Truncated / Broken JSON** | 25 | **Drop** | Incomplete strings cut off mid-generation (missing keys/closing braces) cannot provide supervisory signal. |
| **Exact Duplicate Tickets** | 60 | **Drop** | Identical query text and labels across multiple rows artificially inflate specific classes; dropped to prevent memorization. |
| **Key Typo (`prio` instead of `priority`)** | 70 | **Fix** | Mapped legacy labels (`low` $\rightarrow$ `P3`, `medium` $\rightarrow$ `P2`, `high` $\rightarrow$ `P1`) and renamed key to `priority`. |
| **Date Format Inconsistencies** | 30 | **Fix** | Dates formatted as `DD/MM/YYYY` were normalized to ISO `YYYY-MM-DD`. |
| **Malformed `txn_id`** | 20 | **Fix** | Normalized uncleaned IDs (e.g. `NP-9718716109`, lowercase `np...`) to uppercase alphanumeric `NP[0-9]{10}`. |
| **Fraud Priority Violations** | 7 | **Fix** | Under SCHEMA rules, `fraud` unconditionally mandates priority `P1`; mislabeled `P2`/`P3` rows were corrected. |
| **P1 `needs_human` Violations** | 30 | **Fix** | SCHEMA rule: all `P1` tickets strictly mandate `needs_human = true`; false flags were corrected. |

**Outcome:** 85 rows dropped, 124 rows corrected. The final clean dataset `clean_train.jsonl` contains 1,360 pristine, schema-adherent training rows.

---

## 3. Part B: Baseline Evaluations
Evaluated on `dev.jsonl` (200 gold samples) using `score.py`:

1. **Baseline 1 (Base Model, Zero-Shot Fixed Prompt):**
   - **Exact Match: 0.0%**, **JSON Valid: 0.0%**.
   - *Analysis:* Without instruction fine-tuning, the base model produces conversational preamble (e.g., *"Here is the triage JSON..."*) and markdown fences (` ```json `), which `score.py` strictly rejects.
2. **Baseline 2 (Base Model, Rules Injected into Prompt):**
   - **Exact Match: 0.0%**, **Mean Field Acc: 0.0%**.
   - *Analysis:* Even with complete schema definitions in context, greedy zero-shot generation consistently hallucinates explanations, conversational filler, and fails numerical paise arithmetic without fine-tuned formatting adherence.

---

## 4. Part C: Fine-Tuning Pipeline Architecture
- **Base Model Selection:** `Qwen/Qwen2.5-1.5B-Instruct` was selected over larger alternatives because:
  1. Parameter count (1.54B) allows full LoRA fine-tuning and inference within Colab T4's 15 GB VRAM.
  2. Superior native comprehension of Indian financial terminology and code-mixed Hinglish (`hi-en`).
  3. Non-gated access with fast checkpoint download.
- **LoRA Configuration:** $r=16$, $\alpha=32$, targeting all linear projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Trainable parameters: **18,464,768** (1.18% of model weights, safely below the 50M parameter ceiling).
- **Sequence Length Determination:** Empirical tokenization audit on `clean_train.jsonl` revealed:
  - Median length: 94 tokens; 90th percentile: 135 tokens; 95th percentile: 198 tokens.
  - A small tail of tickets (~2%) contains forwarded email threads spanning up to 2,215 tokens.
  - `max_length = 1024` was chosen to encompass >98% of all samples while maintaining efficient batch throughput on T4.
- **Completion-Only Loss Masking:** Input sequences were structured with the Qwen chat template. All system prompt and customer ticket tokens were masked with label `-100`. Loss was strictly computed on the assistant's JSON tokens.

---

## 5. Part D: Controlled Experiments
Five controlled experiments were conducted, altering a single variable at a time on the T4 runtime:

| Experiment ID | Controlled Change | Dev Exact Match | Dev Mean Field Acc | Peak VRAM | Training Time |
|---|---|:---:|:---:|:---:|:---:|
| `exp1a_raw_data` | Raw noisy data (`train.jsonl`, 4-bit, $r=16$, 2 ep) | 55.5% | 92.1% | 11.46 GB | 75.5 min |
| **`exp1b_clean_data`** | **Clean data (`clean_train.jsonl`, 4-bit, $r=16$, 2 ep)** | **58.5%** | **92.2%** | **11.46 GB** | **72.8 min** |
| `exp2_lora_16bit` | 16-bit LoRA (Clean data, bfloat16, $r=16$, 2 ep) | 58.0% | 92.0% | 14.20 GB | 74.1 min |
| `exp3_rank32` | LoRA Rank $r=32$, $\alpha=64$ (Clean data, 4-bit, 2 ep) | 58.0% | 92.1% | 11.85 GB | 76.4 min |
| `exp4_3epochs` | Epoch ablation: 3 epochs (Clean data, 4-bit, $r=16$) | 57.5% | 91.9% | 11.46 GB | 108.5 min |

### Key Experimental Insights:
1. **Data Cleaning Impact:** `exp1b_clean_data` delivered a **+3.0% boost in Exact Match** over `exp1a_raw_data`. Removing corrupted JSON rows and enforcing strict business consistency directly improved the model's policy adherence.
2. **Quantization:** 4-bit NF4 quantization matches 16-bit LoRA performance while reducing memory pressure from 14.2 GB down to 11.46 GB, preventing CUDA out-of-memory errors on T4.
3. **Rank Capacity:** Increasing rank from 16 to 32 doubled adapter parameters (18.5M $\rightarrow$ 37.0M) without noticeable gain (58.0% vs 58.5%), indicating $r=16$ provides sufficient expressive capacity for JSON schema mapping.
4. **Overfitting at 3 Epochs:** Training for 3 epochs caused slight degradation (57.5%), suggesting 2 epochs is the optimal convergence point for 1,360 samples.

---

## 6. Part E: Error Analysis (Dev Set)
Evaluation of our best model (`exp1b_clean_data`) on `dev.jsonl` produced 83 total mismatches across 200 tickets (41.5% error rate; 58.5% exact match). Systematic failure mode categorization reveals:

### Top Failure Categories & Specific Examples:
1. **Subtle Legal Escalation Recognition (Missed P1 Priority):**
   - *Example (`dv-00003`):* Customer writes: `"...I will go to consumer court"`.
   - *Gold:* `priority: P1` | *Pred:* `priority: P3`.
   - *Root Cause:* The customer embedded the threat colloquially at the end of an IVR query without explicit keywords like "lawyer" or "legal notice".
2. **Disputed Amount Priority Threshold (₹5,000 Boundary):**
   - *Examples (`dv-00005`, `dv-00016`, `dv-00021`, `dv-00023`):* Amounts between ₹5,000 and ₹49,999 in `payment_failed` and `refund` tickets (e.g. ₹49,680, ₹25,647, ₹15,775, ₹23,752) require escalation to `P2`.
   - *Gold:* `priority: P2` | *Pred:* `priority: P3`.
   - *Root Cause:* Small models struggle with implicit numerical range conditioning without explicit reasoning traces.
3. **High-Value Threshold (₹50,000 Boundary for P1 & `needs_human`):**
   - *Examples (`dv-00011`, `dv-00012`, `dv-00019`):* Large amounts (₹84,137, ₹2,37,415, ₹2,59,419) mandate `priority: P1` and `needs_human: true`.
   - *Gold:* `priority: P1, needs_human: true` | *Pred:* `priority: P3, needs_human: false`.
   - *Root Cause:* The model accurately extracted the `amount_paise`, but failed to cascade this value into the downstream priority and human triage flags during single-pass greedy decoding.
4. **Rupee-to-Paise Arithmetic Scaling:**
   - *Examples (`dv-00007`, `dv-00022`):* Customer specifies `₹877` and `Rs. 1677`.
   - *Gold:* `amount_paise: 87700` and `167700` | *Pred:* `amount_paise: 877` and `1677`.
   - *Root Cause:* Occasional omission of the $\times 100$ multiplier when the currency sign (`₹` / `Rs.`) is closely attached to the number.

---

## 7. Next Steps (1-Week Roadmap)
With one additional week, we would implement:
1. **Chain-of-Thought Scratchpad Triage:** Allow the model to generate a compact `<reasoning>` block (extracting raw amount, checking legal threat, determining priority tier) before outputting the final JSON, then stripping the scratchpad.
2. **Targeted Data Augmentation:** Synthesize 300 edge-case tickets focusing explicitly on the ₹5,000 and ₹50,000 priority thresholds, and colloquial legal mentions across English and Hinglish.
3. **Direct Preference Optimization (DPO):** Construct preference pairs where chosen outputs adhere to priority escalation and rejected outputs default to `P3`.

---

## 8. AI Tools Disclosure
- **AI Coding Assistant:** Used for authoring data auditing inspection scripts (`clean_data.py`), generating the notebook scaffolding, and formatting LaTeX tables.
- **Verification:** All data cleaning rules, model training parameters, experimental results, and failure case interpretations were manually reviewed, executed, and verified.

