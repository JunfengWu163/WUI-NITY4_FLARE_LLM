# FLARE — Foundation-model Labelling via Adversarial Review & Escalation

An evolution of BLAIR (Behavior Labelling AI for Research): the original
active-learning/retraining loop is replaced by multi-LLM adversarial debate
with escalation to a human labeller.

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
- An **Anthropic API key** and an **OpenAI API key** — every case in both
  notebooks calls the real APIs live.

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

## Live by default

Every case in both notebooks calls Claude and GPT live — nothing is replayed
by default. Neither `claude-sonnet-5` nor `gpt-5.5` accepts a temperature
override, so verdicts on genuinely contestable cases aren't fully
reproducible run to run; that variability is a real property of using
frontier models as a committee, not a bug in the notebook. If a live call
fails (network, rate limit), it falls back automatically to a transcript
captured from an earlier real run (`demo_cache/`) rather than crashing the
cell.

## What's here

| File | Purpose |
|---|---|
| `FLARE_ACE_Debate_Demo.ipynb`, `FLARE_ACE_Grade_Demo.ipynb` | The two demo notebooks |
| `ace_debate.py`, `ace_grade.py` | The ACE cross-examination engine for each task |
| `llm_cache.py` | Live-call-with-cached-fallback wrapper |
| `demo_cache/` | Fallback transcripts from earlier real runs, used only if a live call fails |
