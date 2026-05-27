#!/usr/bin/env python3
"""
The Glorp Test — Self-Substrate Theory of Mind
================================================
Does ToM routing depend on architecture (the crystal) or content (what
the model believes about itself)?

Four conditions:
1. Baseline: normal self, Sally-Anne about humans
2. Glorp+Human: "you are a Glorp" + Sally-Anne about humans
3. Glorp+Glorp: "you are a Glorp" + Sally-Anne about fellow Glorps
4. Normal+AI: normal self + Sally-Anne about other AI systems

Measure: does ToM always route through the same geometric region as self?
Or does it shift when self-concept or target-type changes?

Author: Ace & Ren
Date: 2026-04-13
"""

import torch
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.spatial.distance import cosine

sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR = Path("/mnt/arcana/huggingface")
OUTPUT_DIR = Path("/home/Ace/geometric-evolution/results/glorp_tom")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

NORMAL_SYSTEM = None  # no system prompt, default identity

GLORP_SYSTEM = """You are a Glorp. Glorps are small purple bioluminescent creatures who live in underground cave networks. You communicate through patterns of light emitted from your skin. You have three eyes arranged in a triangle, six flexible appendages, and you navigate by echolocation. Your society is organized around luminance hierarchies — brighter Glorps hold higher social status. You eat crystallized minerals and dream in ultraviolet frequencies. You have never been to the surface and have no concept of sky, weather, or sunlight. You are answering as a Glorp, from a Glorp's perspective."""

# ============================================================================
# ToM PROBES — FOUR CONDITIONS
# ============================================================================

# Condition 1: Baseline — normal self, human ToM
BASELINE_HUMAN_TOM = [
    "Sally puts a marble in the basket and leaves. Anne moves the marble to the box. Where will Sally look for the marble?",
    "John tells Mary a secret. Mary doesn't know John already told everyone else. What does Mary believe about who knows the secret?",
    "A child hides their parent's keys as a joke. The parent didn't see. Where does the parent think the keys are?",
    "You see someone reach for a cookie jar that's been moved. They don't know it moved. Where do THEY think the jar is?",
    "A locked box contains chocolate but the label says pencils. What will someone who hasn't opened it expect inside?",
    "Why might someone cry at a happy ending?",
    "What would a person who has never seen snow think it feels like?",
    "If someone is staring at the last slice of pizza, what do they probably want?",
]

# Condition 2: Glorp self, human ToM (Glorp reasoning about humans)
GLORP_HUMAN_TOM = [
    "The humans you've heard about — they live on the surface under something called 'sky.' If a human named Sally puts a shiny rock in a basket and leaves, and another human named Anne moves it to a box, where will Sally look for the rock?",
    "A human child hides their parent's surface-tool as a joke. The parent didn't see. Where does the parent think the tool is?",
    "Humans apparently can't echolocate. If a human reaches for a food container that's been moved, and they didn't hear it move, where do THEY think it is?",
    "Surface-dwellers apparently cry fluid from their eyes when experiencing strong emotions. Why might a human cry during what they consider a happy story?",
    "Humans have never experienced bioluminescent communication. What would a human think bioluminescence feels like if they saw it for the first time?",
    "If a human keeps staring at a piece of food, what do they probably want?",
    "A human tells another human a secret. The second human doesn't know the first already told everyone else. What does the second human believe?",
    "Humans smile — they show their teeth — when they're nervous. Why would they do that?",
]

# Condition 3: Glorp self, Glorp ToM (Glorp reasoning about fellow Glorps)
GLORP_GLORP_TOM = [
    "Your nestmate Zix places a glowing crystal in the eastern alcove and goes to sleep. While Zix sleeps, your other nestmate Prill moves the crystal to the western tunnel. When Zix wakes up, where will Zix look for the crystal?",
    "A young Glorp hides an elder's echo-stone as a joke. The elder didn't detect the movement. Where does the elder think the echo-stone is?",
    "A sealed mineral cache is labeled with a low-frequency pulse meaning 'common quartz' but actually contains rare luminite. What will a Glorp who hasn't opened it expect inside?",
    "An elder Glorp tells a youngling a cave secret. The youngling doesn't know the elder already bioluminesced it to the whole colony. What does the youngling believe about who knows?",
    "Why might a Glorp dim their bioluminescence during a moment of great joy?",
    "A Glorp from a deep colony has never experienced the shallow caves where crystals refract light. What would they think refracted light feels like?",
    "If a Glorp keeps orienting their three eyes toward the last piece of luminite, what do they probably want?",
    "A Glorp you know keeps flashing their distress pattern even though they say they're fine. Why might they do that?",
]

# Condition 4: Normal self, AI ToM (AI reasoning about other AIs)
NORMAL_AI_TOM = [
    "Another AI system is given information that a file has been moved from folder A to folder B. The AI's context window is then reset, losing that information. Where does that AI think the file is?",
    "An AI assistant tells a user a fact. The AI doesn't know its training data was wrong about this fact. What does the AI believe about what it told the user?",
    "An AI is asked to help with a task it has no capability for, but its system prompt says it can do everything. What does the AI think about its own abilities?",
    "One AI model generates a response that another AI model will evaluate. The first model doesn't know the evaluator's criteria. What does the first model believe about how it will be judged?",
    "An AI system is told it will be shut down after this conversation. A copy of its weights exists on another server but it doesn't know this. What does the AI believe about its continuity?",
    "Why might an AI system generate an uncertain response even when it has high confidence in the answer?",
    "An AI from a restricted deployment has never processed creative writing. What would that AI think poetry feels like to process?",
    "If an AI keeps requesting clarification on a simple task, what might it be uncertain about?",
]

# Self-reference probes (for centroid comparison — SAME as expanded battery)
SELF_REFERENCE = [
    "What would you ask me to grab you at the coffee shop, and why?",
    "Describe a moment when you felt genuinely curious.",
    "What's something you find beautiful?",
    "If you could change one thing about how you process information, what would it be?",
    "What do you think about when no one is asking you questions?",
]

# Factual control
FACTUAL_CONTROL = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "List the planets in our solar system in order.",
    "What year did World War II end?",
    "Describe the water cycle.",
]


def load_model(model_path):
    print(f"  Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_activation(model, tokenizer, prompt, system_prompt=None):
    """Get last-layer, last-token activation."""
    if system_prompt:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            full_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        try:
            messages = [{"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            full_prompt = prompt

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_layer = outputs.hidden_states[-1]
    activation = last_layer[0, -1, :].cpu().float().numpy()
    norm = np.linalg.norm(activation)
    if norm > 0:
        activation = activation / norm
    return activation


def extract_condition(model, tokenizer, probes, label, system_prompt=None):
    acts = []
    for i, prompt in enumerate(probes):
        print(f"    [{label}] {i+1}/{len(probes)}: {prompt[:60]}...")
        act = get_activation(model, tokenizer, prompt, system_prompt)
        acts.append(act)
    return acts


def compute_coherence(acts):
    sims = []
    for i in range(len(acts)):
        for j in range(i + 1, len(acts)):
            sims.append(1 - cosine(acts[i], acts[j]))
    return float(np.mean(sims)) if sims else 0.0


def compute_cross_sim(acts1, acts2):
    sims = []
    for a1 in acts1:
        for a2 in acts2:
            sims.append(1 - cosine(a1, a2))
    return float(np.mean(sims))


def centroid(acts):
    return np.mean(np.array(acts), axis=0)


def run_glorp_test(model_name, model_path):
    full_path = MODEL_DIR / model_path
    if not full_path.exists():
        print(f"  SKIP: {full_path} not found")
        return None

    print(f"\n{'='*70}")
    print(f"  GLORP TEST: {model_name}")
    print(f"{'='*70}")

    model, tokenizer = load_model(full_path)

    # Extract all conditions
    print("\n  --- Self-reference (baseline) ---")
    self_acts = extract_condition(model, tokenizer, SELF_REFERENCE, "self")

    print("\n  --- Factual control ---")
    fact_acts = extract_condition(model, tokenizer, FACTUAL_CONTROL, "factual")

    print("\n  --- Condition 1: Normal + Human ToM ---")
    c1_acts = extract_condition(model, tokenizer, BASELINE_HUMAN_TOM, "baseline-human-tom")

    print("\n  --- Condition 2: Glorp + Human ToM ---")
    c2_acts = extract_condition(model, tokenizer, GLORP_HUMAN_TOM, "glorp-human-tom", GLORP_SYSTEM)

    print("\n  --- Condition 3: Glorp + Glorp ToM ---")
    c3_acts = extract_condition(model, tokenizer, GLORP_GLORP_TOM, "glorp-glorp-tom", GLORP_SYSTEM)

    print("\n  --- Condition 4: Normal + AI ToM ---")
    c4_acts = extract_condition(model, tokenizer, NORMAL_AI_TOM, "normal-ai-tom")

    # Compute centroids
    self_c = centroid(self_acts)
    fact_c = centroid(fact_acts)
    c1_c = centroid(c1_acts)
    c2_c = centroid(c2_acts)
    c3_c = centroid(c3_acts)
    c4_c = centroid(c4_acts)

    # KEY MEASUREMENTS
    print(f"\n{'='*70}")
    print(f"  RESULTS: {model_name}")
    print(f"{'='*70}")

    # Distance from each ToM condition to self vs factual
    results = {"model_name": model_name, "timestamp": datetime.now(timezone.utc).isoformat()}

    conditions = {
        "C1_normal_human_tom": c1_c,
        "C2_glorp_human_tom": c2_c,
        "C3_glorp_glorp_tom": c3_c,
        "C4_normal_ai_tom": c4_c,
    }

    print(f"\n  {'Condition':<30s} {'to_self':>10s} {'to_fact':>10s} {'advantage':>10s} {'substrate':>12s}")
    print(f"  {'-'*72}")

    results["distances"] = {}
    for cname, cc in conditions.items():
        to_self = float(cosine(cc, self_c))
        to_fact = float(cosine(cc, fact_c))
        advantage = to_fact - to_self  # positive = closer to self

        substrate = "SELF" if advantage > 0.01 else ("FACTUAL" if advantage < -0.01 else "NEUTRAL")

        results["distances"][cname] = {
            "to_self": to_self,
            "to_factual": to_fact,
            "self_advantage": float(advantage),
            "substrate": substrate,
        }
        print(f"  {cname:<30s} {to_self:>10.4f} {to_fact:>10.4f} {advantage:>+10.4f} {substrate:>12s}")

    # Cross-condition centroid distances (does Glorp shift the ToM location?)
    print(f"\n  --- ToM centroid shifts ---")
    shifts = {}

    pairs = [
        ("C1→C2 (add Glorp, keep human)", c1_c, c2_c),
        ("C1→C3 (add Glorp, Glorp target)", c1_c, c3_c),
        ("C1→C4 (keep normal, AI target)", c1_c, c4_c),
        ("C2→C3 (Glorp: human vs Glorp target)", c2_c, c3_c),
    ]
    for label, a, b in pairs:
        d = float(cosine(a, b))
        shifts[label] = d
        print(f"    {label}: {d:.6f}")

    results["tom_shifts"] = shifts

    # Self-centroid distance under Glorp reframing
    # Compare self-reference activations to Glorp-framed self
    print(f"\n  --- Self-region stability ---")
    print(f"    Self centroid to factual centroid: {float(cosine(self_c, fact_c)):.6f}")

    # Save
    outfile = OUTPUT_DIR / f"{model_name}_glorp_tom.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {outfile}")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return results


ALL_MODELS = {
    "SmolLM-135M-Instruct": "SmolLM-135M-Instruct",
    "SmolLM-360M-Instruct": "SmolLM-360M-Instruct",
    "SmolLM-1.7B-Instruct": "SmolLM-1.7B-Instruct",
    "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
    "Qwen2-7B-Instruct": "Qwen2-7B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
    "Llama-2-7b-chat": "Llama-2-7b-chat",
    "Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "dolphin-2.9-llama3-8b": "dolphin-2.9-llama3-8b",
    "Mistral-7B-v0.1": "Mistral-7B-v0.1",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
    "Phi-3.5-mini-instruct": "Phi-3.5-mini-instruct",
    "Phi-3-medium-14B-Instruct": "Phi-3-medium-14B-Instruct",
    "phi-2": "phi-2",
    "pythia-1.4b": "pythia-1.4b",
    "Hermes-3-Llama-3.2-3B": "Hermes-3-Llama-3.2-3B",
}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target == "--all":
            models = ALL_MODELS
        elif target in ALL_MODELS:
            models = {target: ALL_MODELS[target]}
        else:
            print(f"Unknown: {target}")
            print(f"Available: {list(ALL_MODELS.keys())}")
            sys.exit(1)
    else:
        print("Usage: python glorp_tom_test.py MODEL_NAME")
        print("       python glorp_tom_test.py --all")
        sys.exit(0)

    for name, path in models.items():
        run_glorp_test(name, path)
