# LinkedIn Job Description Generator

Generation of job posting descriptions from structured attributes (title, location, skills, work type, experience level, industry), comparing three approaches with very different levels of "intelligence".

## The three models compared

1. **FLAN-T5-small zero-shot** — Google's pretrained model used as-is, with no training on LinkedIn data. Baseline that shows what pretraining alone provides.
2. **FLAN-T5-small fine-tuned** — the same model, fine-tuned on the cleaned LinkedIn dataset. Shows how much domain fine-tuning adds.
3. **Character Transformer from scratch** — a small character-level Transformer decoder trained from scratch. Control experiment showing what happens without pretraining, with the same architecture as the classic Tiny Shakespeare exercise.

**Why FLAN-T5-small?** It's an encoder-decoder Transformer pretrained by Google on a large mixture of instruction-following tasks. The "small" version (~77M parameters) fits in 6 GB of VRAM.

## Repository structure

```
.
├── project.ipynb              # Main notebook with the full pipeline
├── app.py                     # Flask web UI to compare the three models
├── requirements.txt           # Python dependencies
├── data/                      # Raw CSVs + cleaned cache (to download)
├── checkpoint/                # Fine-tuned FLAN-T5 weights (to download)
└── checkpoint_scratch/        # Char Transformer weights + vocabulary (to download)
```

> **Note:** the notebook `project.ipynb` was written by me as part of the project. The file `app.py`, on the other hand, was generated with AI assistance: it's just a small interactive demo to visually compare the outputs of the three models, not part of the original work.

## Data and pre-trained checkpoints

The heavy files (raw Kaggle CSVs, the `df_cleaned.csv` cache, and the model weights) are not in the repo because they're too large. They're available at this Google Drive link:

🔗 **[https://drive.google.com/file/d/1vQS7SPNK6qX2cJps26IQ8UsEe0a3sAmY/view?usp=sharing]**

Once downloaded, place the folders at the root of the project so the structure matches the one described above.

If you want to redo everything from scratch (regenerate cache + train models) just put the raw CSVs in `data/` and run the notebook — the `checkpoint/` and `checkpoint_scratch/` folders will be created automatically.

## Setup

Install the requirements and execute the app.py file to see examples, to train or modify the project notebook you need to execute everything to tokenize and clean the examples (also setup the training method, like CUDA specifications)

## Usage

### Notebook

Open `project.ipynb` and run the cells in order. The notebook is divided into 6 sections:

1. **Setup** — imports, seeds, parameters.
2. **Data cleaning** — starting from the raw CSVs, filters and cleans. Saves a cache in `data/df_cleaned.csv` to avoid re-running the (slow) pipeline every time.
3. **Prompt and target construction** — structured template for input, cleaned description for output. 80/10/10 split into train/val/test.
4. **Tokenization** — HuggingFace `AutoTokenizer`, padding to `max_length`, the `-100` trick on padding labels.
5. **Fine-tuning FLAN-T5** — AdamW with warmup + linear decay, weight decay 0.01, gradient clipping. Saves the best checkpoint based on val loss. Time: ~6-10 hours on a GTX 1660 Super.
6. **Char Transformer** — Transformer decoder trained from scratch on ~1M characters. Vocabulary of ~90 unique characters, context 256, 10 epochs.
7. **Final comparison** — generates the same prompt with all three models and shows the outputs side by side.

If you've already downloaded the checkpoints from the Drive link, you can skip Sections 5 and 6 (training) and go straight to the comparison.

### Web interface

Once the checkpoints are in place:

```bash
python app.py
```

Open `http://localhost:5000` in your browser. Fill in the form with title, location, skills, etc. and see the three outputs side by side.

## Technical notes

**Hardware tested:** GTX 1660 Super (6 GB VRAM), Windows + WSL2.

**Train/serve parity:** the `build_prompt` function is duplicated identically in the notebook and in `app.py`. Same input template at training and inference time, otherwise quality degrades silently.

**Generation parameters:**
- `max_length=384` (FLAN-T5-small's pretraining limit)
- `num_beams=4` (beam search instead of greedy)
- `no_repeat_ngram_size=3` (T5 tends to repeat itself, this prevents it)
- `early_stopping=True`

**Metric:** cross-entropy loss on validation and test. Test loss is typically lower because dropout is disabled in evaluation. No generation-specific metrics like BLEU or ROUGE were implemented.

## Expected results

For the same prompt, qualitatively very different outputs:

- **Zero-shot:** one or two generic sentences. Understands the prompt but doesn't know what a LinkedIn job description should look like.
- **Fine-tuned:** complete and structured description, with recognizable sections like title/role/qualifications/responsibilities.
- **Char Transformer:** gibberish that has the shape of English words but no real vocabulary or meaning.

The comparison concretely shows the value of:
1. Having a pretrained model (zero-shot is already decent vs. char Transformer being unusable).
2. Adding domain fine-tuning (specialization to the "job posting" genre).

## Main dependencies

- **PyTorch** + **torch.nn** — tensors, autograd, training loop.
- **transformers** (HuggingFace) — `T5ForConditionalGeneration`, `AutoTokenizer`, scheduler with warmup.
- **pandas / numpy** — CSV loading and data manipulation.
- **scikit-learn** — train/val/test split.
- **langdetect** — English-only filter on descriptions.
- **flask** — web interface for the interactive comparison.
- **tqdm** — progress bars.

Full list in `requirements.txt`.
