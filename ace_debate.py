"""
FLARE — Adversarial Cross-Examination (ACE) Debate Engine
==========================================================
Bidirectional adversarial debate between Claude and GPT-5 for
social media post relevance labelling in evacuation behaviour mining.

Three-round mechanism:
  Round 1 — Independent verdicts (Claude and GPT each analyse the post alone)
  Round 2 — Adversarial flaw-finding (each model critiques the other's reasoning)
  Round 3 — Rebuttal and convergence (each model may revise; verdict is determined)

Outcome:
  - AGREE     → auto-label, no human needed
  - CONVERGED → auto-label with lower confidence, flag for optional review
  - DISAGREE  → escalate to human with both arguments and flaw reports

Usage:
    Ensure .env is configured, then:
        from ace_debate import ACEDebate
        engine = ACEDebate()
        result = engine.debate(post_text)
        print(result)
"""

import os
import json
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import anthropic
import openai

from llm_cache import call_with_cache, parse_json_response, CACHE_DIR

load_dotenv()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    RELEVANT   = "relevant"
    IRRELEVANT = "irrelevant"
    UNSURE     = "unsure"


class DebateOutcome(str, Enum):
    AGREE      = "agree"       # Both reached the same verdict independently
    CONVERGED  = "converged"   # Disagreed in R1 but converged after flaw-finding
    DISAGREE   = "disagree"    # Still disagreed after R3 — escalate to human


@dataclass
class Round1Position:
    """A model's independent verdict after reading the post."""
    model_name: str
    verdict: Verdict
    evidence: str       # Specific textual evidence cited
    reasoning: str      # Chain-of-thought explanation


@dataclass
class Round2FlawReport:
    """A model's adversarial critique of the other model's Round 1 argument."""
    critic_model: str
    target_model: str
    flaws_found: list[str]   # List of specific logical flaws or missing evidence
    overall_strength: str    # "strong" | "weak" | "contradictory"


@dataclass
class Round3Rebuttal:
    """A model's revised verdict after seeing the flaw report against it."""
    model_name: str
    revised_verdict: Verdict
    revised_reasoning: str
    changed_mind: bool


@dataclass
class ACEResult:
    """Full result of one ACE debate round."""
    post_text: str
    outcome: DebateOutcome
    final_verdict: Verdict | None    # None if DISAGREE (human decides)
    confidence: float                # 0.0–1.0
    round1_claude: Round1Position
    round1_gpt: Round1Position
    round2_claude_flaw: Round2FlawReport   # Claude critiques GPT
    round2_gpt_flaw: Round2FlawReport      # GPT critiques Claude
    round3_claude: Round3Rebuttal
    round3_gpt: Round3Rebuttal
    human_escalation_summary: str | None   # Populated only when outcome == DISAGREE


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ROUND1_PROMPT = """You are analysing a social media post about a wildfire event.

Your task: determine whether this post is relevant to the POST AUTHOR'S OWN evacuation.

A post is relevant if:
- The author personally evacuated, is evacuating, or decided not to evacuate
- The author describes their own evacuation destination, route, or circumstances
- The author's own evacuation decision is influenced by specific factors

A post is NOT relevant if:
- It reports others' evacuations without describing the author's own
- It is a news report or third-party account
- It mentions evacuation in passing without the author being directly involved

Post:
\"\"\"
{post_text}
\"\"\"

Respond in JSON with exactly these fields:
{{
  "verdict": "relevant" | "irrelevant" | "unsure",
  "evidence": "<quote or describe the specific text that most supports your verdict>",
  "reasoning": "<step-by-step reasoning chain explaining your conclusion>"
}}"""


ROUND2_FLAW_PROMPT = """You are acting as an adversarial reviewer.

The following argument was made about a social media post. Your job is to find every
flaw in this argument — logical errors, unsupported claims, missing evidence, or
alternative interpretations that undermine its conclusion.

Be specific. Quote the post text or the argument text directly when identifying flaws.

Post:
\"\"\"
{post_text}
\"\"\"

Argument to critique (verdict: {target_verdict}):
Evidence cited: {target_evidence}
Reasoning: {target_reasoning}

Respond in JSON with exactly these fields:
{{
  "flaws_found": ["<flaw 1>", "<flaw 2>", ...],
  "overall_strength": "strong" | "weak" | "contradictory"
}}

If you genuinely cannot find meaningful flaws, say so — do not invent flaws."""


ROUND3_REBUTTAL_PROMPT = """You previously analysed a social media post and reached a verdict.
Another model found the following flaws in your reasoning.

Post:
\"\"\"
{post_text}
\"\"\"

Your original verdict: {original_verdict}
Your original reasoning: {original_reasoning}

Flaws identified in your argument:
{flaw_list}

Consider these flaws carefully. Do they change your conclusion?

Respond in JSON with exactly these fields:
{{
  "revised_verdict": "relevant" | "irrelevant" | "unsure",
  "revised_reasoning": "<updated reasoning, addressing the flaws raised>",
  "changed_mind": true | false
}}"""


# ---------------------------------------------------------------------------
# ACE Debate Engine
# ---------------------------------------------------------------------------

class ACEDebate:

    def __init__(self):
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=90.0,
            max_retries=0,
        )
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=90.0,
            max_retries=0,
        )
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-5.5")

    # --- Low-level API calls (each wrapped for cached fallback) ---------------

    def _ask_claude(self, prompt: str, cache_key: str) -> dict:
        def live():
            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}]
            )
            text_block = next(b for b in response.content if b.type == "text")
            return parse_json_response(text_block.text)
        return call_with_cache(cache_key, live)

    def _ask_gpt(self, prompt: str, cache_key: str) -> dict:
        def live():
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                max_completion_tokens=8192,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}]
            )
            return parse_json_response(response.choices[0].message.content)
        return call_with_cache(cache_key, live)

    # --- Round implementations ------------------------------------------------

    def _round1(self, post_text: str, case_id: str) -> tuple[Round1Position, Round1Position]:
        """Both models independently analyse the post."""
        prompt = ROUND1_PROMPT.format(post_text=post_text)

        claude_raw = self._ask_claude(prompt, f"{case_id}_r1_claude")
        gpt_raw    = self._ask_gpt(prompt, f"{case_id}_r1_gpt")

        claude_pos = Round1Position(
            model_name="claude",
            verdict=Verdict(claude_raw["verdict"]),
            evidence=claude_raw["evidence"],
            reasoning=claude_raw["reasoning"]
        )
        gpt_pos = Round1Position(
            model_name="gpt",
            verdict=Verdict(gpt_raw["verdict"]),
            evidence=gpt_raw["evidence"],
            reasoning=gpt_raw["reasoning"]
        )
        return claude_pos, gpt_pos

    def _round2(
        self,
        post_text: str,
        claude_pos: Round1Position,
        gpt_pos: Round1Position,
        case_id: str
    ) -> tuple[Round2FlawReport, Round2FlawReport]:
        """Each model finds flaws in the other's Round 1 argument."""

        # Claude critiques GPT
        claude_critique_prompt = ROUND2_FLAW_PROMPT.format(
            post_text=post_text,
            target_verdict=gpt_pos.verdict,
            target_evidence=gpt_pos.evidence,
            target_reasoning=gpt_pos.reasoning
        )
        claude_flaw_raw = self._ask_claude(claude_critique_prompt, f"{case_id}_r2_claude")
        claude_flaw = Round2FlawReport(
            critic_model="claude",
            target_model="gpt",
            flaws_found=claude_flaw_raw["flaws_found"],
            overall_strength=claude_flaw_raw["overall_strength"]
        )

        # GPT critiques Claude
        gpt_critique_prompt = ROUND2_FLAW_PROMPT.format(
            post_text=post_text,
            target_verdict=claude_pos.verdict,
            target_evidence=claude_pos.evidence,
            target_reasoning=claude_pos.reasoning
        )
        gpt_flaw_raw = self._ask_gpt(gpt_critique_prompt, f"{case_id}_r2_gpt")
        gpt_flaw = Round2FlawReport(
            critic_model="gpt",
            target_model="claude",
            flaws_found=gpt_flaw_raw["flaws_found"],
            overall_strength=gpt_flaw_raw["overall_strength"]
        )

        return claude_flaw, gpt_flaw

    def _round3(
        self,
        post_text: str,
        claude_pos: Round1Position,
        gpt_pos: Round1Position,
        claude_received_flaw: Round2FlawReport,   # flaws GPT found in Claude
        gpt_received_flaw: Round2FlawReport,        # flaws Claude found in GPT
        case_id: str
    ) -> tuple[Round3Rebuttal, Round3Rebuttal]:
        """Each model revises its verdict in light of flaw reports against it."""

        # Claude responds to GPT's flaws
        claude_rebuttal_prompt = ROUND3_REBUTTAL_PROMPT.format(
            post_text=post_text,
            original_verdict=claude_pos.verdict,
            original_reasoning=claude_pos.reasoning,
            flaw_list="\n".join(f"- {f}" for f in claude_received_flaw.flaws_found)
        )
        claude_reb_raw = self._ask_claude(claude_rebuttal_prompt, f"{case_id}_r3_claude")
        claude_rebuttal = Round3Rebuttal(
            model_name="claude",
            revised_verdict=Verdict(claude_reb_raw["revised_verdict"]),
            revised_reasoning=claude_reb_raw["revised_reasoning"],
            changed_mind=claude_reb_raw["changed_mind"]
        )

        # GPT responds to Claude's flaws
        gpt_rebuttal_prompt = ROUND3_REBUTTAL_PROMPT.format(
            post_text=post_text,
            original_verdict=gpt_pos.verdict,
            original_reasoning=gpt_pos.reasoning,
            flaw_list="\n".join(f"- {f}" for f in gpt_received_flaw.flaws_found)
        )
        gpt_reb_raw = self._ask_gpt(gpt_rebuttal_prompt, f"{case_id}_r3_gpt")
        gpt_rebuttal = Round3Rebuttal(
            model_name="gpt",
            revised_verdict=Verdict(gpt_reb_raw["revised_verdict"]),
            revised_reasoning=gpt_reb_raw["revised_reasoning"],
            changed_mind=gpt_reb_raw["changed_mind"]
        )

        return claude_rebuttal, gpt_rebuttal

    # --- Verdict resolution ---------------------------------------------------

    def _resolve(
        self,
        r1_claude: Round1Position,
        r1_gpt: Round1Position,
        r3_claude: Round3Rebuttal,
        r3_gpt: Round3Rebuttal
    ) -> tuple[DebateOutcome, Verdict | None, float]:
        """Determine the final outcome and confidence score."""

        # Agreed immediately in Round 1
        if r1_claude.verdict == r1_gpt.verdict and r1_claude.verdict != Verdict.UNSURE:
            return DebateOutcome.AGREE, r1_claude.verdict, 0.95

        # Converged after flaw-finding
        if r3_claude.revised_verdict == r3_gpt.revised_verdict and \
                r3_claude.revised_verdict != Verdict.UNSURE:
            return DebateOutcome.CONVERGED, r3_claude.revised_verdict, 0.75

        # Still in disagreement — escalate
        return DebateOutcome.DISAGREE, None, 0.0

    def _escalation_summary(
        self,
        post_text: str,
        r1_claude: Round1Position,
        r1_gpt: Round1Position,
        r2_claude_flaw: Round2FlawReport,
        r2_gpt_flaw: Round2FlawReport,
        r3_claude: Round3Rebuttal,
        r3_gpt: Round3Rebuttal
    ) -> str:
        """Generate a structured summary to help the human labeller decide quickly."""
        lines = [
            "=== HUMAN REVIEW REQUIRED ===",
            f"Post: {post_text[:200]}{'...' if len(post_text) > 200 else ''}",
            "",
            f"Claude final verdict: {r3_claude.revised_verdict}",
            f"  Reasoning: {r3_claude.revised_reasoning}",
            "",
            f"GPT final verdict: {r3_gpt.revised_verdict}",
            f"  Reasoning: {r3_gpt.revised_reasoning}",
            "",
            "Flaws Claude found in GPT's argument:",
            *[f"  - {f}" for f in r2_claude_flaw.flaws_found],
            "",
            "Flaws GPT found in Claude's argument:",
            *[f"  - {f}" for f in r2_gpt_flaw.flaws_found],
            "",
            "Label this post T (relevant) or F (irrelevant):"
        ]
        return "\n".join(lines)

    # --- Pinned-transcript replay -----------------------------------------------

    def debate_from_pinned_transcript(self, post_text: str, case_id: str) -> ACEResult:
        """Reconstruct a result from a permanently pinned transcript — no API calls.

        See ACEGrade.grade_from_pinned_transcript for why this exists: model
        verdicts aren't perfectly deterministic across calls, and for a case
        whose point is to show a specific outcome live, replaying a real
        captured transcript is more reliable than hoping a live call
        reproduces it.
        """
        pinned_dir = CACHE_DIR / "pinned"

        def load(round_model: str) -> dict:
            data = json.loads((pinned_dir / f"{case_id}_{round_model}.json").read_text())
            data.pop("_source", None)
            return data

        r1_claude_raw, r1_gpt_raw = load("r1_claude"), load("r1_gpt")
        r1_claude = Round1Position("claude", Verdict(r1_claude_raw["verdict"]),
                                    r1_claude_raw["evidence"], r1_claude_raw["reasoning"])
        r1_gpt = Round1Position("gpt", Verdict(r1_gpt_raw["verdict"]),
                                 r1_gpt_raw["evidence"], r1_gpt_raw["reasoning"])

        if r1_claude.verdict == r1_gpt.verdict and r1_claude.verdict != Verdict.UNSURE:
            placeholder_flaw = Round2FlawReport("n/a", "n/a", [], "strong")
            placeholder_reb = Round3Rebuttal("n/a", r1_claude.verdict, "agreed in R1", False)
            return ACEResult(
                post_text=post_text,
                outcome=DebateOutcome.AGREE, final_verdict=r1_claude.verdict, confidence=0.95,
                round1_claude=r1_claude, round1_gpt=r1_gpt,
                round2_claude_flaw=placeholder_flaw, round2_gpt_flaw=placeholder_flaw,
                round3_claude=placeholder_reb, round3_gpt=placeholder_reb,
                human_escalation_summary=None
            )

        r2_claude_raw, r2_gpt_raw = load("r2_claude"), load("r2_gpt")
        r2_claude_flaw = Round2FlawReport("claude", "gpt",
                                           r2_claude_raw["flaws_found"], r2_claude_raw["overall_strength"])
        r2_gpt_flaw = Round2FlawReport("gpt", "claude",
                                        r2_gpt_raw["flaws_found"], r2_gpt_raw["overall_strength"])

        r3_claude_raw, r3_gpt_raw = load("r3_claude"), load("r3_gpt")
        r3_claude = Round3Rebuttal("claude", Verdict(r3_claude_raw["revised_verdict"]),
                                    r3_claude_raw["revised_reasoning"], r3_claude_raw["changed_mind"])
        r3_gpt = Round3Rebuttal("gpt", Verdict(r3_gpt_raw["revised_verdict"]),
                                 r3_gpt_raw["revised_reasoning"], r3_gpt_raw["changed_mind"])

        outcome, final_verdict, confidence = self._resolve(r1_claude, r1_gpt, r3_claude, r3_gpt)

        escalation = None
        if outcome == DebateOutcome.DISAGREE:
            escalation = self._escalation_summary(
                post_text, r1_claude, r1_gpt, r2_claude_flaw, r2_gpt_flaw, r3_claude, r3_gpt
            )

        return ACEResult(
            post_text=post_text,
            outcome=outcome, final_verdict=final_verdict, confidence=confidence,
            round1_claude=r1_claude, round1_gpt=r1_gpt,
            round2_claude_flaw=r2_claude_flaw, round2_gpt_flaw=r2_gpt_flaw,
            round3_claude=r3_claude, round3_gpt=r3_gpt,
            human_escalation_summary=escalation
        )

    # --- Main entry point -----------------------------------------------------

    def debate(self, post_text: str, case_id: str) -> ACEResult:
        """Run the full three-round ACE debate for one social media post.

        case_id must be unique per post — it namespaces the cached-fallback
        responses for this case.
        """

        print(f"  Round 1: independent verdicts...", end=" ", flush=True)
        r1_claude, r1_gpt = self._round1(post_text, case_id)
        print(f"Claude={r1_claude.verdict}, GPT={r1_gpt.verdict}")

        # Early exit: both agree immediately
        if r1_claude.verdict == r1_gpt.verdict and r1_claude.verdict != Verdict.UNSURE:
            print(f"  Agreed in Round 1 — no debate needed.")
            # Placeholder rebuttal objects for data consistency
            placeholder_flaw = Round2FlawReport("n/a", "n/a", [], "strong")
            placeholder_reb  = Round3Rebuttal("n/a", r1_claude.verdict, "agreed in R1", False)
            return ACEResult(
                post_text=post_text,
                outcome=DebateOutcome.AGREE,
                final_verdict=r1_claude.verdict,
                confidence=0.95,
                round1_claude=r1_claude, round1_gpt=r1_gpt,
                round2_claude_flaw=placeholder_flaw, round2_gpt_flaw=placeholder_flaw,
                round3_claude=placeholder_reb, round3_gpt=placeholder_reb,
                human_escalation_summary=None
            )

        print(f"  Round 2: adversarial flaw-finding...", end=" ", flush=True)
        r2_claude_flaw, r2_gpt_flaw = self._round2(post_text, r1_claude, r1_gpt, case_id)
        print(f"Claude found {len(r2_claude_flaw.flaws_found)} flaw(s) in GPT, "
              f"GPT found {len(r2_gpt_flaw.flaws_found)} flaw(s) in Claude")

        print(f"  Round 3: rebuttals...", end=" ", flush=True)
        r3_claude, r3_gpt = self._round3(
            post_text, r1_claude, r1_gpt,
            claude_received_flaw=r2_gpt_flaw,   # Claude receives GPT's flaws
            gpt_received_flaw=r2_claude_flaw,      # GPT receives Claude's flaws
            case_id=case_id
        )
        print(f"Claude={r3_claude.revised_verdict} (changed={r3_claude.changed_mind}), "
              f"GPT={r3_gpt.revised_verdict} (changed={r3_gpt.changed_mind})")

        outcome, final_verdict, confidence = self._resolve(r1_claude, r1_gpt, r3_claude, r3_gpt)

        escalation = None
        if outcome == DebateOutcome.DISAGREE:
            escalation = self._escalation_summary(
                post_text, r1_claude, r1_gpt,
                r2_claude_flaw, r2_gpt_flaw, r3_claude, r3_gpt
            )

        return ACEResult(
            post_text=post_text,
            outcome=outcome,
            final_verdict=final_verdict,
            confidence=confidence,
            round1_claude=r1_claude, round1_gpt=r1_gpt,
            round2_claude_flaw=r2_claude_flaw, round2_gpt_flaw=r2_gpt_flaw,
            round3_claude=r3_claude, round3_gpt=r3_gpt,
            human_escalation_summary=escalation
        )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Two test posts from the Blair paper (representative cases)
    test_posts = [
        # Clear relevance — should AGREE quickly
        "Finally made it to CPVenturaBeach. Checked in, grabbed a margarita at the bar "
        "& made it JUST in time #TickFire",

        # Ambiguous — likely to trigger full debate
        "I'm in Canyon Country, my power just went out. Anyone else out here after "
        "evacuating too? Did your power go out? @someone #KincadeFire",
    ]

    engine = ACEDebate()

    for i, post in enumerate(test_posts, 1):
        print(f"\n{'='*60}")
        print(f"Post {i}: {post[:80]}...")
        print(f"{'='*60}")
        result = engine.debate(post, case_id=f"quick_demo_{i}")
        print(f"\nOutcome:  {result.outcome}")
        print(f"Verdict:  {result.final_verdict}")
        print(f"Confidence: {result.confidence:.0%}")
        if result.human_escalation_summary:
            print()
            print(result.human_escalation_summary)
