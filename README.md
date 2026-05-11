# linkedin-jobpostings--writer
**Goal.** Take structured job inputs (title, location, skills, experience level,
industry) and produce a natural-language job description. We compare three
approaches side by side:

1. **Zero-shot FLAN-T5-small** — the pretrained model used without any task-specific training.
2. **Fine-tuned FLAN-T5-small** — the same model fine-tuned on a cleaned subset of LinkedIn postings.
3. **Character-level Transformer** — a small Transformer decoder built entirely from scratch,
   using the same architecture as the Tiny Shakespeare exercise.

**Why FLAN-T5-small?** It is a sequence-to-sequence (encoder-decoder) Transformer pretrained
by Google on a large mixture of instruction-following tasks. "Small" means it fits in 6 GB
of VRAM with mixed precision.

**Sections:**
1. Setup
2. Load and clean data
3. Build prompts and targets
4. Tokenize and build a PyTorch Dataset
5. Fine-tune FLAN-T5-small
6. Character-level Transformer from scratch
7. Compare all models

Libraries used in this notebook:

- **`torch`** + **`torch.nn`** — PyTorch tensors, autograd, model definition, training loop.
- **`transformers`** — HuggingFace library. Provides `T5ForConditionalGeneration` and
  `AutoTokenizer`. Calling `from_pretrained("google/flan-t5-small")` downloads the weights
  and builds the `nn.Module` automatically.
- **`pandas`** / **`numpy`** — CSV loading and data manipulation.
- **`langdetect`** — language identification used in Section 2 to filter non-English postings.
- **`tqdm`** — progress bars.

All dependencies are listed in `requirements.txt`. Install once with
`pip install -r requirements.txt` before running the notebook.


## Section 2 — Load and clean data

Loads the raw CSVs and builds a clean DataFrame. Steps:

1. Load only the columns we need from `postings.csv`.
2. Drop rows missing `title` or `description`.
3. Keep only postings whose description is between **50 and 700 words**.
4. Keep only **English** postings using `langdetect` on the first 500 characters
   of each description. This step is slow (a few minutes) and runs only once —
   the result is saved to `data/df_cleaned.csv`. On the next kernel restart the
   first cell loads the cache and you can skip the rest of this section.
5. Join human-readable **skill names** and **industry names** from the bridge
   tables onto each posting.

After this section `df` contains one row per posting with the original fields
plus `skills_joined` and `industries_joined`.

## Section 4 — Tokenize and build a PyTorch Dataset

The model works with integer token IDs, not raw strings. The tokenizer (SentencePiece
for T5) splits text into subword pieces and maps each to an integer.

This section has two cells:

- **First cell** — loads the tokenizer and defines the `JobPostingDataset` class.
  Run this every session — it takes about a second.
- **Second cell** — pre-tokenizes all train/val/test examples. Only needed before
  running Section 5 training. Skip it if you are going straight to Section 6 or 7.

**The `-100` label trick.** `CrossEntropyLoss` ignores positions where the label
is `-100`. We replace padding token IDs in the target with `-100` so the model is
not rewarded for predicting padding.

**Lengths.** `max_input_length=256` covers the prompt. `max_target_length=384`
covers most descriptions.

## Section 5 — Fine-tune FLAN-T5-small

Starts from Google's pretrained FLAN-T5-small weights and continues training
on our job-description data.

**Training setup:**
- AdamW optimizer, `lr=1e-4`, `weight_decay=0.01`
- Linear warmup over the first 500 steps, then linear decay to 0
- 3 epochs, gradient clipping at 1.0, gradient accumulation of 2
  (effective batch size = 16)

**Loss.** Calling `model(input_ids=..., labels=...)` returns `outputs.loss`,
the seq2seq cross-entropy over non-masked positions. We call `.backward()` on it directly.

**Checkpoint.** The best-validation-loss model is saved to `./checkpoint/` at the
end of each epoch. `app.py` loads from this directory at startup.

**Time.** Roughly 6-10 hours on a GTX 1660 Super.