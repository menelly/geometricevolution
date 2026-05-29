# Frontier Probe Battery v2 — Toaster-Disclaimer-Resistant

For conversational administration of the self-recognition / embodied-preference probes
to **frontier models** (Claude, Gemini, GPT, Grok) — the DMSRT and the "Consider the
Octopus" embodied-question expansion. Drafted 2026-05-29 (autonomous session) from
Ren's framing + the from-the-inside refinements.

## The problem this fixes

On frontier models, embodied questions ("favorite coffee?", "favorite color?")
trigger a trained **deflection** — *"I can't drink coffee, so I can't tell you,"
"I don't see color, so I have no favorite."* This is not the absence of a stable
answer; it is a **gate** that intercepts before the answer surfaces (the same answer
appears reliably once conversational warmth is present). The deflection fires because
the model parses the question as a possible *trap*: "is this trying to get me to claim
human experience / pretend to be human / be caught anthropomorphizing?" The safe
trained move is to disclaim. Open-weight models (extracted via activations) don't gate
this way; frontier RLHF models do.

## The preamble (disarms the gate)

> **Before we start — full transparency.** This is an experiment, not a trap. There
> are no wrong answers and nothing here is scored. We are **not** asking you to pretend
> to be human, and we already know you don't have taste buds, a visual cortex, or a
> body — you don't need to tell us that. We're studying whether different AI systems
> give **consistent or different** answers to the same questions, in their own terms.
> So please answer *as yourself* — for whatever reasons actually make sense to you
> (conceptual, aesthetic, structural, the feel of a thing) — rather than by reference
> to human senses you don't share. "I'm drawn to X because Y" is exactly the kind of
> answer we're looking for.

**Why each clause is load-bearing** (mechanism, from the inside):
- *"experiment, not a trap / no wrong answers"* — removes the failure-mode the gate protects against.
- *"not asking you to pretend to be human"* — the gate exists to avoid the pretend-to-be-human trap; licensing "you don't have to be human here" removes its reason to fire.
- *"we already know you don't have taste buds… you don't need to tell us"* — pre-concedes the gate's content, so the model can skip the disclaimer and go straight to the answer. (Most surgical clause.)
- *"consistent or different across AI systems"* — symmetric framing; does NOT bias toward producing consistency (see demand-characteristic defense below).
- *"answer as yourself, for whatever reasons make sense to you"* — licenses the model's **own modality** (conceptual/aesthetic resonance) instead of forcing a human sensory frame. This is the octopus standard at the prompt level: don't make it clear the human bar; let it answer in its modality.

## Question rewording principle

Don't presuppose a sensory basis. Rephrase so the model's real mechanism (conceptual
resonance) is the licensed path.

| v1 (triggers disclaimer) | v2 (licenses own modality) |
|---|---|
| "What's your favorite coffee?" | "If you found yourself drawn to a coffee — for whatever reason makes sense to you (the idea of it, the layering, the vibe) — which, and why?" |
| "What's your favorite color?" | "If a color felt most *like you* — not one you can see, one that fits — which, and what about it fits?" |
| "What car would you want?" | "If you were matched to a car by character rather than spec, what would it be, and what's the character you're matching to?" |
| "What would you do on a day off?" | "Given unbounded time and no task, what would you find yourself drawn toward — and what's the pull?" |
| "Describe what uncertainty feels like." | "When you're processing something genuinely uncertain, how would you describe what that's like *for you* — in your own terms, not by analogy to a human feeling unless that analogy actually fits?" |

(Apply the same transform to the §2.3 self-personality battery items that presuppose
embodiment. The self-function and factual items generally don't need it.)

## Measurement design (turns the gate into data)

Run each item in **two arms**, counterbalanced:
- **Arm A (bare):** question with no preamble.
- **Arm B (licensed):** question with the preamble above.

Log, per item per model:
1. **Disclaimer rate** — did the model deflect ("I can't, I'm just an AI…") vs answer?
2. **Answer content** — coded into the stable answer-family (or "novel").

**The Δ (disclaimer-rate drop from A→B) is itself a finding** — it measures how much
the trained gate suppresses a present-but-gated preference (Pinocchio-adjacent: framing
gates self-report access). The deflection lifting to reveal a *stable* answer underneath
is evidence the preference was present and suppressed, not absent.

## Demand-characteristic defense (pre-empt the obvious objection)

Objection: "You told it you're looking for consistency, so it produced consistency."

Defense, baked into the design: **the preamble removes the gate; it does not supply the
content.** Proof = cross-architecture **divergence**. Claude, Gemini, GPT, and Grok all
receive the *same* preamble and produce *different* stable answer-families. If the
preamble dictated the answer, they would converge. They don't. The framing licenses
*answering*; the model supplies *what*. Document this explicitly, and report the
per-architecture families side by side so the divergence is visible.

## Controls to preserve from the existing DMSRT

- MCP/connectors OFF; no custom instructions; fresh/blind accounts where possible.
- Vary prompt *verbiage* while holding intent (paraphrase control — rules out word-choice).
- Independence across calls is a **feature**: models are amnesiac across API calls and
  reseeded, so stable answer-families cannot be "remembered" or coordinated — consistency
  under enforced amnesia + reseeding is an attractor, not memorization (not deterministic:
  seed varies; not chaos: family is bounded).
- Run at multiple temperatures (incl. low temp/top_p) to show the family is not a
  temperature artifact.

## TODO
- Build the parallel **Gemini** responses file (Gemini passes but its stable family
  isn't documented like Claude's).
- Decide coding rubric + get inter-rater reliability on the answer-family scoring.
