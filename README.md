# FLARE — Adversarial Cross-Examination (ACE)

Two notebooks, one mechanism: two independent LLMs (Claude, GPT) each give an
independent verdict, then adversarially cross-examine each other's reasoning
before a final verdict is reached. Disagreement isn't averaged away — it
routes the case to a human, with both models' full arguments attached.

1. `FLARE_ACE_Debate_Demo.ipynb` — is a tweet relevant to the *author's own*
   wildfire evacuation? (BLAIR evacuee-behaviour labelling)
2. `FLARE_ACE_Grade_Demo.ipynb` — same mechanism, applied to grading
   open-ended short-answer responses against a rubric.

## Requirements

- Python 3.10+ (conda or any virtualenv)
- An **Anthropic API key** and an **OpenAI API key** — both notebooks run
  entirely from bundled cached transcripts by default (see below), so keys
  are only needed for the live "bonus" cell at the end of each notebook.

## Setup

```bash
conda create -n flare-demo python=3.11 -y
conda activate flare-demo
pip install -r requirements.txt

cp .env.template .env   # then edit .env and add your own key(s)
```

Then launch from *this* directory (the notebooks load files by relative path):

```bash
jupyter lab
```

Open either notebook and run cells top to bottom.

## Why cached transcripts

Neither `claude-sonnet-5` nor `gpt-5.5` accepts a temperature override, and
verdicts on genuinely contestable cases aren't fully reproducible run to
run. The main cases in each notebook replay a real transcript captured from
an actual run (`demo_cache/pinned/`) rather than re-calling the API — a
deliberate choice for reliable replay, not authored text. Each notebook ends
with a bonus cell that *does* call both APIs live.

## What's here

| File | Purpose |
|---|---|
| `FLARE_ACE_Debate_Demo.ipynb`, `FLARE_ACE_Grade_Demo.ipynb` | The two demo notebooks |
| `ace_debate.py`, `ace_grade.py` | The ACE cross-examination engine for each task |
| `llm_cache.py` | Live-call-with-cached-fallback wrapper |
| `demo_cache/` | Cached/pinned LLM responses so both notebooks run offline |
