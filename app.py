"""LinkedIn job description generator — multi-model comparison UI.

Run:
    python app.py
Then open http://localhost:5000 in a browser.

Models loaded at startup:
  [1] Zero-shot FLAN-T5-small  — pretrained weights, no fine-tuning
  [2] Fine-tuned FLAN-T5-small — trained in notebook Section 5
  [3] Char-level Transformer   — trained from scratch in notebook Section 6

The char Transformer requires:
  checkpoint_scratch/char_transformer.pt
  checkpoint_scratch/char_vocab.json
Both are written when you run Section 6 of the notebook.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from flask import Flask, request
from transformers import AutoTokenizer, T5ForConditionalGeneration

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR         = Path("./checkpoint")
CHECKPOINT_SCRATCH_DIR = Path("./checkpoint_scratch")
BASE_MODEL_NAME        = "google/flan-t5-small"
MAX_INPUT_LEN          = 256
GEN_KWARGS = dict(max_length=384, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)

# ===========================================================================
# build_prompt — COPY VERBATIM from notebook Section 3.
# ===========================================================================
def _clean_text(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def _or_unspecified(value) -> str:
    if value is None:
        return "not specified"
    if isinstance(value, float) and pd.isna(value):
        return "not specified"
    s = _clean_text(value)
    return s if s else "not specified"

def build_prompt(row) -> str:
    return (
        "generate a job description for the following role.\n"
        f"title: {_or_unspecified(row.get('title'))}\n"
        f"work_type: {_or_unspecified(row.get('formatted_work_type'))}\n"
        f"experience: {_or_unspecified(row.get('formatted_experience_level'))}\n"
        f"location: {_or_unspecified(row.get('location'))}\n"
        f"industry: {_or_unspecified(row.get('industries_joined'))}\n"
        f"skills: {_or_unspecified(row.get('skills_joined'))}"
    )
# ===========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[startup] Device: {DEVICE}")

# ---------------------------------------------------------------------------
# T5 models
# ---------------------------------------------------------------------------
print(f"[startup] Loading fine-tuned model from {CHECKPOINT_DIR} ...")
tokenizer   = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
model_tuned = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_DIR).to(DEVICE).eval()

print(f"[startup] Loading zero-shot baseline {BASE_MODEL_NAME} ...")
model_base = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME).to(DEVICE).eval()

@torch.no_grad()
def run_t5(model, prompt: str) -> str:
    enc = tokenizer(prompt, max_length=MAX_INPUT_LEN, truncation=True, return_tensors="pt").to(DEVICE)
    return tokenizer.decode(model.generate(**enc, **GEN_KWARGS)[0], skip_special_tokens=True)

# ---------------------------------------------------------------------------
# Char-level Transformer (optional — loaded only if checkpoint exists)
# ---------------------------------------------------------------------------
class TransformerEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size, sequence_len):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(sequence_len, embedding_dim)
        self.register_buffer("positions", torch.arange(sequence_len).reshape(1, -1))

    def forward(self, x):
        seq_len = x.shape[1]
        return (self.token_embedding(x)
                + self.position_embedding(self.positions[:, :seq_len]))


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, sequence_len, embedding_dim,
                 n_layer, dropout, mlp_dim, nhead):
        super().__init__()
        self.embedding = TransformerEmbedding(embedding_dim, vocab_size, sequence_len)
        block = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead,
            dim_feedforward=mlp_dim, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=n_layer)
        self.classifier  = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.classifier(x)


TRANSFORMER_PATH = CHECKPOINT_SCRATCH_DIR / "char_transformer.pt"
VOCAB_PATH       = CHECKPOINT_SCRATCH_DIR / "char_vocab.json"

run_char_transformer = None

if TRANSFORMER_PATH.exists() and VOCAB_PATH.exists():
    print(f"[startup] Loading char Transformer from {TRANSFORMER_PATH} ...")
    _vd        = json.loads(VOCAB_PATH.read_text())
    _itos_list = _vd["itos"]
    _stoi      = {c: i for i, c in enumerate(_itos_list)}
    _itos      = {i: c for i, c in enumerate(_itos_list)}
    _VOCAB     = len(_itos_list)
    _CTX       = _vd["context_len"]

    _transf = TransformerDecoder(
        vocab_size=_VOCAB,          sequence_len=_CTX,
        embedding_dim=_vd["embedding_dim"], n_layer=_vd["n_layer"],
        dropout=_vd.get("dropout", 0.2),    mlp_dim=_vd["mlp_dim"],
        nhead=_vd["n_heads"],
    )
    _transf.load_state_dict(torch.load(str(TRANSFORMER_PATH), map_location=DEVICE, weights_only=True))
    _transf.to(DEVICE).eval()

    def _sample_top_p(logits, p=0.9, temperature=1.0):
        temperature = max(temperature, 1e-5)
        scaled = logits / temperature
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        filtered = scaled.clone()
        filtered[remove.scatter(0, sorted_idx, remove)] = float("-inf")
        return torch.multinomial(torch.softmax(filtered, dim=-1), 1)

    @torch.no_grad()
    def run_char_transformer(prompt: str, length: int = 400,
                             temperature: float = 1.0, p: float = 0.9) -> str:
        seed = [_stoi.get(c, 0) for c in prompt if c in _stoi]
        while len(seed) < length:
            context = seed[-_CTX:]
            x = torch.tensor(context, dtype=torch.long, device=DEVICE).reshape(1, -1)
            logits = _transf(x)[0, -1, :].cpu()
            seed.append(_sample_top_p(logits, p=p, temperature=temperature).item())
        return "".join(_itos.get(i, "") for i in seed)

print("[startup] Ready.")

# ---------------------------------------------------------------------------
# UI options
# ---------------------------------------------------------------------------
WORK_TYPES = ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Volunteer", "Other"]
EXP_LEVELS = ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not specified"]

def _options(values, selected):
    return "".join(
        f'<option value="{v}"{"  selected" if v == selected else ""}>{v}</option>'
        for v in values
    )

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Job Description Generator</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1400px;margin:2em auto;padding:0 1em;color:#222}
h1{margin-bottom:.2em}
label{display:block;margin-top:.7em;font-weight:600}
input[type=text],select{width:100%;box-sizing:border-box;padding:.4em;font:inherit}
button{margin-top:1em;padding:.6em 1.4em;font-size:1em;cursor:pointer}
.output{white-space:pre-wrap;background:#f5f5f5;padding:1em;border:1px solid #ddd;
        border-radius:4px;margin-top:.4em;font-size:.93em;line-height:1.5}
.row{display:flex;gap:.6em}.row>*{flex:1}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1em;margin-top:1em}
.col{display:flex;flex-direction:column}
.col h3{margin:.2em 0 .3em;font-size:1em}
.col .output{flex:1}
.tuned      h3{color:#0a6}
.base       h3{color:#888}
.char-transf h3{color:#b60}
.prompt-box{margin-top:1.2em}
</style></head><body>
<h1>Job Description Generator</h1>
<p>Outputs from all available models are shown side by side.</p>
<form method="post">
  <label>Title <input type="text" name="title" value="__TITLE__" required></label>
  <div class="row">
    <label>Work type<select name="work_type">__WORK_TYPE_OPTIONS__</select></label>
    <label>Experience level<select name="experience">__EXPERIENCE_OPTIONS__</select></label>
  </div>
  <label>Location <input type="text" name="location" value="__LOCATION__"></label>
  <label>Industry (comma-separated) <input type="text" name="industry" value="__INDUSTRY__"></label>
  <label>Skills (comma-separated) <input type="text" name="skills" value="__SKILLS__"></label>
  <button type="submit">Generate</button>
</form>
__OUTPUT_BLOCK__
</body></html>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))

def _col(css_class: str, title: str, text: str) -> str:
    return (
        f"<div class='col {css_class}'>"
        f"<h3>{title}</h3>"
        f"<div class='output'>{_escape(text)}</div>"
        f"</div>"
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    form = {
        "title": "", "work_type": "Full-time", "experience": "Mid-Senior level",
        "location": "", "industry": "", "skills": "",
    }
    output_block = ""

    if request.method == "POST":
        form["title"]      = request.form.get("title", "").strip()
        form["work_type"]  = request.form.get("work_type", "Full-time")
        form["experience"] = request.form.get("experience", "Mid-Senior level")
        form["location"]   = request.form.get("location", "").strip()
        form["industry"]   = request.form.get("industry", "").strip()
        form["skills"]     = request.form.get("skills", "").strip()

        row = {
            "title":                      form["title"] or float("nan"),
            "formatted_work_type":        form["work_type"],
            "formatted_experience_level": None if form["experience"] == "Not specified" else form["experience"],
            "location":                   form["location"] or float("nan"),
            "industries_joined":          form["industry"] or float("nan"),
            "skills_joined":              form["skills"]   or float("nan"),
        }
        prompt = build_prompt(row)
        print("=" * 60)
        print("PROMPT:")
        print(prompt)
        print("=" * 60)

        _na = "(model not available — run Section 6 first)"
        cols = ""
        cols += _col("base",        f"[1] Zero-shot ({BASE_MODEL_NAME})",          run_t5(model_base,  prompt))
        cols += _col("tuned",       "[2] Fine-tuned FLAN-T5 (Section 5)",           run_t5(model_tuned, prompt))
        cols += _col("char-transf", "[3] Char Transformer (scratch, Section 6)",
                     run_char_transformer(prompt) if run_char_transformer is not None else _na)

        output_block = (
            "<h2>Generated descriptions</h2>"
            f"<div class='grid'>{cols}</div>"
            "<div class='prompt-box'><strong>Prompt sent to all models:</strong>"
            f"<div class='output'>{_escape(prompt)}</div></div>"
        )

    html = (HTML
        .replace("__TITLE__",              _escape(form["title"]))
        .replace("__WORK_TYPE_OPTIONS__",  _options(WORK_TYPES, form["work_type"]))
        .replace("__EXPERIENCE_OPTIONS__", _options(EXP_LEVELS, form["experience"]))
        .replace("__LOCATION__",           _escape(form["location"]))
        .replace("__INDUSTRY__",           _escape(form["industry"]))
        .replace("__SKILLS__",             _escape(form["skills"]))
        .replace("__OUTPUT_BLOCK__",       output_block)
    )
    return html

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
