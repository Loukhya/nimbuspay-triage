# NimbusPay Support Ticket Triage - LLM Fine-Tuning

Fine-tuned small language model for automated customer support ticket triage into an 8-key structured JSON record under strict business rules.

## Model Summary
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters, open weights)
- **Quantization:** 4-bit NormalFloat (NF4) via BitsAndBytes (`load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.bfloat16`)
- **Fine-Tuning Stack:** PEFT LoRA (QLoRA) with 18.46M trainable parameters (1.18% of total parameters, strictly $\le$ 50M limit)
- **Best Result:** **58.5% Exact Match** and **92.2% Mean Field Accuracy** on `dev.jsonl` (a **+3.0% boost in Exact Match** over the raw dataset baseline).

## Repository Contents
```
├── README.md               # Model specification and summary
├── report.md               # 3-page technical report (audit, experiments, error analysis)
├── cleaning_log.csv        # Log of all 209 dropped/fixed rows in training data
├── notebook.ipynb          # Google Colab T4 reproduction notebook with outputs
├── predictions_test.jsonl  # Predictions on 400 test tickets
└── adapter/                # Final PEFT LoRA adapter
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Inference Protocol
- Greedy decoding (`do_sample=False`), `max_new_tokens=200`.
- System prompt: `"You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."`
- Test input gets only the system prompt and ticket text formatted via Qwen chat template. Zero rules or few-shot examples in prompt at test time.
- Output post-processing: `.strip()` whitespace only.

