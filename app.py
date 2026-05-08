"""LinkedIn job description generator (local web UI).

Run:
    python app.py
Then open http://localhost:5000 in a browser.

The model and tokenizer are loaded once at startup. Form submissions build a
prompt with the EXACT same build_prompt function used during training
(notebook Section 3) and feed it to model.generate(...). The prompt is also
printed to the console every time, so you can visually compare it to the
training prompts shown in the notebook.

You need Flask installed:  pip install flask
"""
import re
from pathlib import Path

import pandas as pd
import torch
from flask import Flask, request
from transformers import AutoTokenizer, T5ForConditionalGeneration

# -----------------------------------------------------------------------------
# Config -- must match notebook Sections 4 and 7.
# -----------------------------------------------------------------------------
CHECKPOINT_DIR   = Path("./checkpoint")     # the fine-tuned FLAN-T5-small
BASE_MODEL_NAME  = "google/flan-t5-small"   # zero-shot baseline (Section 5)
MAX_INPUT_LEN  = 256
GEN_KWARGS = dict(
    max_length=384,
    num_beams=4,
    no_repeat_ngram_size=3,
    early_stopping=True,
)

# =============================================================================
# build_prompt -- COPY VERBATIM from notebook Section 3.
# Any divergence between training and inference prompts silently ruins output.
# =============================================================================
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
# =============================================================================

# -----------------------------------------------------------------------------
# Model load -- once, at startup. Auto-detect CUDA, fall back to CPU.
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[startup] Device: {DEVICE}")
print(f"[startup] Loading fine-tuned model from {CHECKPOINT_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
model_tuned = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_DIR).to(DEVICE).eval()
print(f"[startup] Loading zero-shot baseline {BASE_MODEL_NAME} ...")
model_base = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME).to(DEVICE).eval()
print("[startup] Ready.")

# -----------------------------------------------------------------------------
# UI options. WORK_TYPES / EXP_LEVELS match LinkedIn's vocabulary so the form
# submits values the model actually saw during training.
# -----------------------------------------------------------------------------
WORK_TYPES = ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Volunteer", "Other"]
EXP_LEVELS = ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not specified"]

def _options(values, selected):
    out = []
    for v in values:
        sel = " selected" if v == selected else ""
        out.append(f'<option value="{v}"{sel}>{v}</option>')
    return "".join(out)

# -----------------------------------------------------------------------------
# Inline HTML template. Ugly but functional, single string, no Jinja file.
# -----------------------------------------------------------------------------
HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Job Description Generator</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 1em;color:#222}
h1{margin-bottom:.2em}
label{display:block;margin-top:.7em;font-weight:600}
input[type=text],input[type=number],select{width:100%;box-sizing:border-box;padding:.4em;font:inherit}
button{margin-top:1em;padding:.6em 1.2em;font-size:1em;cursor:pointer}
.output{white-space:pre-wrap;background:#f5f5f5;padding:1em;border:1px solid #ddd;margin-top:.5em;border-radius:4px}
.row{display:flex;gap:.6em}.row>*{flex:1}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:1em;margin-top:1em}
.compare h3{margin:0 0 .3em 0}
.compare .col{display:flex;flex-direction:column}
.compare .col .output{flex:1}
.tuned h3{color:#0a6}
.base h3{color:#888}
@media (max-width:760px){.compare{grid-template-columns:1fr}}
</style></head><body>
<h1>Job Description Generator</h1>
<p>FLAN-T5-small on LinkedIn postings. Outputs from the fine-tuned model and the zero-shot baseline are shown side by side.</p>
<form method="post">
  <label>Title <input type="text" name="title" value="__TITLE__" required></label>
  <div class="row">
    <label>Work type
      <select name="work_type">__WORK_TYPE_OPTIONS__</select>
    </label>
    <label>Experience level
      <select name="experience">__EXPERIENCE_OPTIONS__</select>
    </label>
  </div>
  <label>Location <input type="text" name="location" value="__LOCATION__"></label>
  <label>Industry (comma-separated) <input type="text" name="industry" value="__INDUSTRY__"></label>
  <label>Skills (comma-separated) <input type="text" name="skills" value="__SKILLS__"></label>
  <button type="submit">Generate</button>
</form>
__OUTPUT_BLOCK__
</body></html>
"""

# -----------------------------------------------------------------------------
# Inference.
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_model(model, prompt: str) -> str:
    enc = tokenizer(prompt, max_length=MAX_INPUT_LEN, truncation=True, return_tensors="pt").to(DEVICE)
    out_ids = model.generate(**enc, **GEN_KWARGS)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

def _escape(s: str) -> str:
    """Tiny HTML-escape so user input doesn't break the page."""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))

# -----------------------------------------------------------------------------
# Routes.
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    # Defaults so a GET shows an empty-ish form.
    form = {
        "title": "",
        "work_type": "Full-time",
        "experience": "Mid-Senior level",
        "location": "",
        "industry": "",
        "skills": "",
    }
    output_block = ""

    if request.method == "POST":
        form["title"]      = request.form.get("title", "").strip()
        form["work_type"]  = request.form.get("work_type", "Full-time")
        form["experience"] = request.form.get("experience", "Mid-Senior level")
        form["location"]   = request.form.get("location", "").strip()
        form["industry"]   = request.form.get("industry", "").strip()
        form["skills"]     = request.form.get("skills", "").strip()

        # Build a row dict shaped like a training-time df row, so build_prompt
        # works without modification. Empty strings become NaN so they map to
        # "not specified" inside _or_unspecified.
        row = {
            "title":                      form["title"] or float("nan"),
            "formatted_work_type":        form["work_type"],
            "formatted_experience_level": None if form["experience"] == "Not specified" else form["experience"],
            "location":                   form["location"] or float("nan"),
            "industries_joined":          form["industry"] or float("nan"),
            "skills_joined":              form["skills"]   or float("nan"),
        }

        prompt = build_prompt(row)
        # Print to console so you can visually compare against training prompts.
        print("=" * 60)
        print("PROMPT SENT TO MODELS:")
        print(prompt)
        print("=" * 60)

        gen_tuned = run_model(model_tuned, prompt)
        gen_base  = run_model(model_base,  prompt)
        output_block = (
            "<h2>Generated descriptions</h2>"
            "<div class='compare'>"
              "<div class='col tuned'>"
                "<h3>Fine-tuned</h3>"
                f"<div class='output'>{_escape(gen_tuned)}</div>"
              "</div>"
              "<div class='col base'>"
                f"<h3>Zero-shot ({_escape(BASE_MODEL_NAME)})</h3>"
                f"<div class='output'>{_escape(gen_base)}</div>"
              "</div>"
            "</div>"
            "<h3>Prompt sent to both models</h3>"
            f"<div class='output'>{_escape(prompt)}</div>"
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
    # Local only; debug=False so the model isn't loaded twice by the reloader.
    app.run(host="127.0.0.1", port=5000, debug=False)
