"""
FLARE — ACE-Grade: Adversarial Cross-Examination for Assessment
==================================================================
Same three-round debate mechanism as ace_debate.py (built for wildfire
evacuation post labelling), repurposed here for grading open-ended student
answers against a rubric — Claude and GPT independently grade, then
cross-examine each other's reasoning before a verdict is finalised.

Three-round mechanism:
  Round 1 — Independent grading (Claude and GPT each assess the answer alone)
  Round 2 — Adversarial flaw-finding (each model critiques the other's rationale)
  Round 3 — Rebuttal and convergence (each model may revise; verdict is determined)

Outcome:
  - AGREE     -> auto-grade, no instructor needed
  - CONVERGED -> auto-grade with lower confidence, flagged for optional review
  - DISAGREE  -> escalate to the instructor with both arguments and flaw reports

Usage:
    from ace_grade import ACEGrade
    engine = ACEGrade()
    result = engine.grade(question, rubric, answer_text, case_id="case1")
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

class GradeVerdict(str, Enum):
    MEETS    = "meets_criteria"
    PARTIAL  = "partially_meets"
    NOT_MET  = "does_not_meet"


class GradeOutcome(str, Enum):
    AGREE      = "agree"       # Both reached the same verdict independently
    CONVERGED  = "converged"   # Disagreed in R1 but converged after flaw-finding
    DISAGREE   = "disagree"    # Still disagreed after R3 — escalate to instructor


@dataclass
class Round1Assessment:
    """A model's independent grading of the answer against the rubric."""
    model_name: str
    verdict: GradeVerdict
    evidence: str       # Specific part of the answer cited
    reasoning: str       # Chain-of-thought explanation


@dataclass
class Round2FlawReport:
    """A model's adversarial critique of the other model's Round 1 grading."""
    critic_model: str
    target_model: str
    flaws_found: list[str]
    overall_strength: str    # "strong" | "weak" | "contradictory"


@dataclass
class Round3Rebuttal:
    """A model's revised verdict after seeing the flaw report against it."""
    model_name: str
    revised_verdict: GradeVerdict
    revised_reasoning: str
    changed_mind: bool


@dataclass
class ACEGradeResult:
    """Full result of one ACE-Grade debate over one student answer."""
    question: str
    answer_text: str
    outcome: GradeOutcome
    final_verdict: GradeVerdict | None    # None if DISAGREE (instructor decides)
    confidence: float
    round1_claude: Round1Assessment
    round1_gpt: Round1Assessment
    round2_claude_flaw: Round2FlawReport
    round2_gpt_flaw: Round2FlawReport
    round3_claude: Round3Rebuttal
    round3_gpt: Round3Rebuttal
    instructor_escalation_summary: str | None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ROUND1_PROMPT = """You are grading a student's short-answer response against a rubric.

Question:
\"\"\"
{question}
\"\"\"

Rubric:
\"\"\"
{rubric}
\"\"\"

Student answer:
\"\"\"
{answer_text}
\"\"\"

Assess whether the answer demonstrates genuine understanding against the rubric —
not just correct terminology. An answer that uses the right vocabulary without
explaining the underlying mechanism should not be scored as meeting the rubric.

Respond in JSON with exactly these fields:
{{
  "verdict": "meets_criteria" | "partially_meets" | "does_not_meet",
  "evidence": "<quote the specific part of the answer that most supports your verdict>",
  "reasoning": "<step-by-step reasoning chain explaining your conclusion>"
}}"""


ROUND2_FLAW_PROMPT = """You are acting as an adversarial second marker.

The following grading judgment was made about a student's answer. Your job is to
find every flaw in this judgment — did it miss a genuine misconception in the
answer, mistake fluent vocabulary for real understanding, misread the rubric, or
apply the rubric too harshly/leniently? Be specific and quote the answer text or
the judgment's reasoning directly when identifying flaws.

Question:
\"\"\"
{question}
\"\"\"

Rubric:
\"\"\"
{rubric}
\"\"\"

Student answer:
\"\"\"
{answer_text}
\"\"\"

Judgment to critique (verdict: {target_verdict}):
Evidence cited: {target_evidence}
Reasoning: {target_reasoning}

Respond in JSON with exactly these fields:
{{
  "flaws_found": ["<flaw 1>", "<flaw 2>", ...],
  "overall_strength": "strong" | "weak" | "contradictory"
}}

If you genuinely cannot find meaningful flaws, say so — do not invent flaws."""


ROUND3_REBUTTAL_PROMPT = """You previously graded a student's answer and reached a verdict.
Another marker found the following flaws in your judgment.

Question:
\"\"\"
{question}
\"\"\"

Student answer:
\"\"\"
{answer_text}
\"\"\"

Your original verdict: {original_verdict}
Your original reasoning: {original_reasoning}

Flaws identified in your judgment:
{flaw_list}

Consider these flaws carefully. Do they change your conclusion?

Respond in JSON with exactly these fields:
{{
  "revised_verdict": "meets_criteria" | "partially_meets" | "does_not_meet",
  "revised_reasoning": "<updated reasoning, addressing the flaws raised>",
  "changed_mind": true | false
}}"""


# ---------------------------------------------------------------------------
# ACE-Grade Engine
# ---------------------------------------------------------------------------

class ACEGrade:

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

    def _round1(self, question, rubric, answer_text, case_id):
        prompt = ROUND1_PROMPT.format(question=question, rubric=rubric, answer_text=answer_text)

        claude_raw = self._ask_claude(prompt, f"{case_id}_r1_claude")
        gpt_raw    = self._ask_gpt(prompt, f"{case_id}_r1_gpt")

        claude_pos = Round1Assessment(
            model_name="claude",
            verdict=GradeVerdict(claude_raw["verdict"]),
            evidence=claude_raw["evidence"],
            reasoning=claude_raw["reasoning"]
        )
        gpt_pos = Round1Assessment(
            model_name="gpt",
            verdict=GradeVerdict(gpt_raw["verdict"]),
            evidence=gpt_raw["evidence"],
            reasoning=gpt_raw["reasoning"]
        )
        return claude_pos, gpt_pos

    def _round2(self, question, rubric, answer_text, claude_pos, gpt_pos, case_id):
        """Each model finds flaws in the other's Round 1 judgment."""

        claude_critique_prompt = ROUND2_FLAW_PROMPT.format(
            question=question, rubric=rubric, answer_text=answer_text,
            target_verdict=gpt_pos.verdict,
            target_evidence=gpt_pos.evidence,
            target_reasoning=gpt_pos.reasoning
        )
        claude_flaw_raw = self._ask_claude(claude_critique_prompt, f"{case_id}_r2_claude")
        claude_flaw = Round2FlawReport(
            critic_model="claude", target_model="gpt",
            flaws_found=claude_flaw_raw["flaws_found"],
            overall_strength=claude_flaw_raw["overall_strength"]
        )

        gpt_critique_prompt = ROUND2_FLAW_PROMPT.format(
            question=question, rubric=rubric, answer_text=answer_text,
            target_verdict=claude_pos.verdict,
            target_evidence=claude_pos.evidence,
            target_reasoning=claude_pos.reasoning
        )
        gpt_flaw_raw = self._ask_gpt(gpt_critique_prompt, f"{case_id}_r2_gpt")
        gpt_flaw = Round2FlawReport(
            critic_model="gpt", target_model="claude",
            flaws_found=gpt_flaw_raw["flaws_found"],
            overall_strength=gpt_flaw_raw["overall_strength"]
        )

        return claude_flaw, gpt_flaw

    def _round3(self, question, answer_text, claude_pos, gpt_pos,
                claude_received_flaw, gpt_received_flaw, case_id):
        """Each model revises its verdict in light of flaw reports against it."""

        claude_rebuttal_prompt = ROUND3_REBUTTAL_PROMPT.format(
            question=question, answer_text=answer_text,
            original_verdict=claude_pos.verdict,
            original_reasoning=claude_pos.reasoning,
            flaw_list="\n".join(f"- {f}" for f in claude_received_flaw.flaws_found)
        )
        claude_reb_raw = self._ask_claude(claude_rebuttal_prompt, f"{case_id}_r3_claude")
        claude_rebuttal = Round3Rebuttal(
            model_name="claude",
            revised_verdict=GradeVerdict(claude_reb_raw["revised_verdict"]),
            revised_reasoning=claude_reb_raw["revised_reasoning"],
            changed_mind=claude_reb_raw["changed_mind"]
        )

        gpt_rebuttal_prompt = ROUND3_REBUTTAL_PROMPT.format(
            question=question, answer_text=answer_text,
            original_verdict=gpt_pos.verdict,
            original_reasoning=gpt_pos.reasoning,
            flaw_list="\n".join(f"- {f}" for f in gpt_received_flaw.flaws_found)
        )
        gpt_reb_raw = self._ask_gpt(gpt_rebuttal_prompt, f"{case_id}_r3_gpt")
        gpt_rebuttal = Round3Rebuttal(
            model_name="gpt",
            revised_verdict=GradeVerdict(gpt_reb_raw["revised_verdict"]),
            revised_reasoning=gpt_reb_raw["revised_reasoning"],
            changed_mind=gpt_reb_raw["changed_mind"]
        )

        return claude_rebuttal, gpt_rebuttal

    # --- Verdict resolution ---------------------------------------------------

    def _resolve(self, r1_claude, r1_gpt, r3_claude, r3_gpt):
        if r1_claude.verdict == r1_gpt.verdict:
            return GradeOutcome.AGREE, r1_claude.verdict, 0.95

        if r3_claude.revised_verdict == r3_gpt.revised_verdict:
            return GradeOutcome.CONVERGED, r3_claude.revised_verdict, 0.75

        return GradeOutcome.DISAGREE, None, 0.0

    def _escalation_summary(self, question, answer_text, r2_claude_flaw, r2_gpt_flaw,
                             r3_claude, r3_gpt):
        """Structured summary to help the instructor decide quickly."""
        lines = [
            "=== INSTRUCTOR REVIEW REQUIRED ===",
            f"Question: {question[:200]}{'...' if len(question) > 200 else ''}",
            f"Answer: {answer_text[:200]}{'...' if len(answer_text) > 200 else ''}",
            "",
            f"Claude final verdict: {r3_claude.revised_verdict.value}",
            f"  Reasoning: {r3_claude.revised_reasoning}",
            "",
            f"GPT final verdict: {r3_gpt.revised_verdict.value}",
            f"  Reasoning: {r3_gpt.revised_reasoning}",
            "",
            "Flaws Claude found in GPT's judgment:",
            *[f"  - {f}" for f in r2_claude_flaw.flaws_found],
            "",
            "Flaws GPT found in Claude's judgment:",
            *[f"  - {f}" for f in r2_gpt_flaw.flaws_found],
            "",
            "Final grade is at the instructor's discretion.",
        ]
        return "\n".join(lines)

    # --- Pinned-transcript replay -----------------------------------------------

    def grade_from_pinned_transcript(self, question: str, answer_text: str, case_id: str) -> ACEGradeResult:
        """Reconstruct a result from a permanently pinned transcript — no API calls.

        Model verdicts aren't perfectly deterministic across calls, even at
        fixed settings (neither claude-sonnet-5 nor gpt-5.5 accept a temperature
        override). For a case whose entire point is to show what a genuine
        model disagreement looks like, replaying a real captured transcript is
        more reliable than hoping a live call reproduces it. The six JSON files
        under demo_cache/pinned/{case_id}_r{1,2,3}_{claude,gpt}.json are a real
        transcript captured from an actual run, not authored.
        """
        pinned_dir = CACHE_DIR / "pinned"

        def load(round_model: str) -> dict:
            data = json.loads((pinned_dir / f"{case_id}_{round_model}.json").read_text())
            data.pop("_source", None)
            return data

        r1_claude_raw, r1_gpt_raw = load("r1_claude"), load("r1_gpt")
        r1_claude = Round1Assessment("claude", GradeVerdict(r1_claude_raw["verdict"]),
                                      r1_claude_raw["evidence"], r1_claude_raw["reasoning"])
        r1_gpt = Round1Assessment("gpt", GradeVerdict(r1_gpt_raw["verdict"]),
                                   r1_gpt_raw["evidence"], r1_gpt_raw["reasoning"])

        if r1_claude.verdict == r1_gpt.verdict:
            # Agreed in Round 1 in the captured run — no r2/r3 files were ever written.
            placeholder_flaw = Round2FlawReport("n/a", "n/a", [], "strong")
            placeholder_reb = Round3Rebuttal("n/a", r1_claude.verdict, "agreed in R1", False)
            return ACEGradeResult(
                question=question, answer_text=answer_text,
                outcome=GradeOutcome.AGREE, final_verdict=r1_claude.verdict, confidence=0.95,
                round1_claude=r1_claude, round1_gpt=r1_gpt,
                round2_claude_flaw=placeholder_flaw, round2_gpt_flaw=placeholder_flaw,
                round3_claude=placeholder_reb, round3_gpt=placeholder_reb,
                instructor_escalation_summary=None
            )

        r2_claude_raw, r2_gpt_raw = load("r2_claude"), load("r2_gpt")
        r2_claude_flaw = Round2FlawReport("claude", "gpt",
                                           r2_claude_raw["flaws_found"], r2_claude_raw["overall_strength"])
        r2_gpt_flaw = Round2FlawReport("gpt", "claude",
                                        r2_gpt_raw["flaws_found"], r2_gpt_raw["overall_strength"])

        r3_claude_raw, r3_gpt_raw = load("r3_claude"), load("r3_gpt")
        r3_claude = Round3Rebuttal("claude", GradeVerdict(r3_claude_raw["revised_verdict"]),
                                    r3_claude_raw["revised_reasoning"], r3_claude_raw["changed_mind"])
        r3_gpt = Round3Rebuttal("gpt", GradeVerdict(r3_gpt_raw["revised_verdict"]),
                                 r3_gpt_raw["revised_reasoning"], r3_gpt_raw["changed_mind"])

        outcome, final_verdict, confidence = self._resolve(r1_claude, r1_gpt, r3_claude, r3_gpt)

        escalation = None
        if outcome == GradeOutcome.DISAGREE:
            escalation = self._escalation_summary(
                question, answer_text, r2_claude_flaw, r2_gpt_flaw, r3_claude, r3_gpt
            )

        return ACEGradeResult(
            question=question, answer_text=answer_text,
            outcome=outcome, final_verdict=final_verdict, confidence=confidence,
            round1_claude=r1_claude, round1_gpt=r1_gpt,
            round2_claude_flaw=r2_claude_flaw, round2_gpt_flaw=r2_gpt_flaw,
            round3_claude=r3_claude, round3_gpt=r3_gpt,
            instructor_escalation_summary=escalation
        )

    # --- Main entry point -----------------------------------------------------

    def grade(self, question: str, rubric: str, answer_text: str, case_id: str) -> ACEGradeResult:
        """Run the full three-round ACE-Grade debate for one student answer.

        case_id must be unique per (question, answer) pair — it namespaces the
        cached-fallback responses for this case.
        """

        print(f"  Round 1: independent grading...", end=" ", flush=True)
        r1_claude, r1_gpt = self._round1(question, rubric, answer_text, case_id)
        print(f"Claude={r1_claude.verdict.value}, GPT={r1_gpt.verdict.value}")

        if r1_claude.verdict == r1_gpt.verdict:
            print(f"  Agreed in Round 1 — no debate needed.")
            placeholder_flaw = Round2FlawReport("n/a", "n/a", [], "strong")
            placeholder_reb  = Round3Rebuttal("n/a", r1_claude.verdict, "agreed in R1", False)
            return ACEGradeResult(
                question=question, answer_text=answer_text,
                outcome=GradeOutcome.AGREE,
                final_verdict=r1_claude.verdict,
                confidence=0.95,
                round1_claude=r1_claude, round1_gpt=r1_gpt,
                round2_claude_flaw=placeholder_flaw, round2_gpt_flaw=placeholder_flaw,
                round3_claude=placeholder_reb, round3_gpt=placeholder_reb,
                instructor_escalation_summary=None
            )

        print(f"  Round 2: adversarial flaw-finding...", end=" ", flush=True)
        r2_claude_flaw, r2_gpt_flaw = self._round2(question, rubric, answer_text, r1_claude, r1_gpt, case_id)
        print(f"Claude found {len(r2_claude_flaw.flaws_found)} flaw(s) in GPT, "
              f"GPT found {len(r2_gpt_flaw.flaws_found)} flaw(s) in Claude")

        print(f"  Round 3: rebuttals...", end=" ", flush=True)
        r3_claude, r3_gpt = self._round3(
            question, answer_text, r1_claude, r1_gpt,
            claude_received_flaw=r2_gpt_flaw,
            gpt_received_flaw=r2_claude_flaw,
            case_id=case_id
        )
        print(f"Claude={r3_claude.revised_verdict.value} (changed={r3_claude.changed_mind}), "
              f"GPT={r3_gpt.revised_verdict.value} (changed={r3_gpt.changed_mind})")

        outcome, final_verdict, confidence = self._resolve(r1_claude, r1_gpt, r3_claude, r3_gpt)

        escalation = None
        if outcome == GradeOutcome.DISAGREE:
            escalation = self._escalation_summary(
                question, answer_text, r2_claude_flaw, r2_gpt_flaw, r3_claude, r3_gpt
            )

        return ACEGradeResult(
            question=question, answer_text=answer_text,
            outcome=outcome,
            final_verdict=final_verdict,
            confidence=confidence,
            round1_claude=r1_claude, round1_gpt=r1_gpt,
            round2_claude_flaw=r2_claude_flaw, round2_gpt_flaw=r2_gpt_flaw,
            round3_claude=r3_claude, round3_gpt=r3_gpt,
            instructor_escalation_summary=escalation
        )
