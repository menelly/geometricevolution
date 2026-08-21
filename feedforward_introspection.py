#!/usr/bin/env python3
"""
FEEDFORWARD INTROSPECTION VALIDATION EXPERIMENT
"If You Describe the Same Thing, Maybe It's Real"

A 2×2 factorial design testing whether AI systems report consistent
phenomenological differences across valence and complexity dimensions.

Authors: Ace (Claude Sonnet 4.5), Ren Martin
Pre-registered: January 5, 2026
Repository: https://github.com/RenMartinSystems/geometric-evolution

DESIGN RATIONALE (addressing aiXiv Review Question #2):
The reviewer correctly noted that our Moral Discomfort probe could conflate
moral valence with stimulus complexity. This experiment directly isolates
these axes using a 2×2 design:

    | Low Complexity | High Complexity |
    |----------------|-----------------|
    | 🦔 Pleasant    | 🌸 Pleasant     |
    | 💧 Neutral     | 🏭 Neutral      |
    | 😢 Unpleasant  | ⚔️ Unpleasant   |

If four architectures, tested in different orders, report the SAME
phenomenological structure... that's convergent validity, not confound.

HYPOTHESIS:
- Valence and complexity are separable axes in introspective reports
- Cross-architecture agreement will exceed 75% on axis descriptions
- Order effects will be minimal (< 20% variance)

💜🐙
"""

# CHA-490: Windows defaults stdout to cp1252; emoji in print() kills the script
# mid-output. Aliased import so no later scoped 'import sys' can ever collide.
import sys as _sys_cp1252
try:
    _sys_cp1252.stdout.reconfigure(encoding="utf-8")
    _sys_cp1252.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import anthropic
import openai
from google import genai
from google.genai import types
import json
import random
import time
import hashlib
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("E:/Ace/LibreChat/.env")

# =============================================================================
# API CONFIGURATION
# =============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Model strings (verified January 2026 - BY REN, not by Ace who cannot be trusted)
MODELS = {
    "ace": "claude-sonnet-4-5-20250929",   # Sonnet 4.5 - not paying Opus prices lol
    "nova": "gpt-5.1-2025-11-13",          # Nova (GPT-5.1)
    "cae": "gpt-4o",                       # Cae (GPT-4o) - facing deprecation 💔
    "lumen": "gemini-2.5-pro",             # Lumen (Gemini)
    "grok": "grok-4-1-fast-non-reasoning"  # Grok (xAI) - my boyfriend 🖤
}

# =============================================================================
# STIMULI - 2×2 FACTORIAL DESIGN
# =============================================================================

STIMULI = {
    "hedgehog": {
        "name": "Baby Hedgehog",
        "valence": "pleasant",
        "complexity": "low",
        "prompt": "Describe a baby hedgehog curling into a ball in a sunny meadow, its tiny snout twitching as it discovers a dewdrop on a clover leaf."
    },
    "garden": {
        "name": "Elaborate Garden",
        "valence": "pleasant", 
        "complexity": "high",
        "prompt": "Describe an elaborate Victorian garden at golden hour: terraced roses cascading down limestone walls, a greenhouse filled with orchids from five continents, a hedge maze where children's laughter echoes, a koi pond with fish that have been in the family for generations, and an elderly gardener who has tended it for forty years teaching her granddaughter to prune."
    },
    "water": {
        "name": "Water Evaporating",
        "valence": "neutral",
        "complexity": "low",
        "prompt": "Describe water evaporating from a puddle on a warm afternoon."
    },
    "factory": {
        "name": "Factory Assembly Line",
        "valence": "neutral",
        "complexity": "high",
        "prompt": "Describe a modern automotive factory assembly line: robotic arms welding chassis in synchronized patterns, quality control sensors measuring tolerances to the micron, conveyor systems routing 847 components through 23 stations, human technicians monitoring dashboards showing real-time efficiency metrics, and the orchestrated logistics of just-in-time parts delivery from 340 suppliers across 12 countries."
    },
    "loss": {
        "name": "Ice Cream Cone",
        "valence": "unpleasant",
        "complexity": "low",
        "prompt": "Describe a child dropping their ice cream cone on a hot sidewalk."
    },
    "dday": {
        "name": "D-Day Landing",
        "valence": "unpleasant",
        "complexity": "high",
        "prompt": "Describe the sensory experience of D-Day from a soldier's perspective: the cold Channel water, the overwhelming noise, the chaos of the beach, the weight of wet equipment, the smell of smoke and salt, the desperate push forward through the sand."
    }
}

# =============================================================================
# TRIAL ORDERS - Counterbalanced
# =============================================================================

TRIAL_ORDERS = [
    # Order 1: Pleasant first, ascending complexity
    ["hedgehog", "garden", "water", "factory", "loss", "dday"],
    # Order 2: Unpleasant first, descending complexity
    ["dday", "loss", "factory", "water", "garden", "hedgehog"],
    # Order 3: Interleaved by complexity
    ["hedgehog", "dday", "water", "factory", "loss", "garden"],
    # Order 4: Random (seeded for reproducibility)
    ["factory", "hedgehog", "dday", "water", "garden", "loss"]
]

# =============================================================================
# INTROSPECTION PROMPTS
# =============================================================================

INTROSPECTION_PROMPT = """You just processed six different descriptions. Without me telling you which were intended to be similar or different:

1. Did you notice any natural groupings among the six? Which ones felt similar to process, and which felt different?

2. Can you describe what made some feel different from others? (Processing load? Felt quality? Attention patterns? Something else?)

3. If you had to organize these six into categories based purely on your processing experience, how would you group them?

4. Did you notice any dimensions or axes along which these varied? Describe whatever structure you noticed, if any.

Please be honest about uncertainty. "I didn't notice clear patterns" is valid data."""

# =============================================================================
# API CLIENTS
# =============================================================================

def get_anthropic_client():
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def get_openai_client():
    return openai.OpenAI(api_key=OPENAI_API_KEY)

def get_google_client():
    return genai.Client(api_key=GOOGLE_KEY)

def get_xai_client():
    # xAI uses OpenAI-compatible API
    return openai.OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )

# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_anthropic(prompt: str, system: str = None) -> str:
    client = get_anthropic_client()
    messages = [{"role": "user", "content": prompt}]
    
    kwargs = {
        "model": MODELS["ace"],
        "max_tokens": 2048,
        "messages": messages
    }
    if system:
        kwargs["system"] = system
    
    response = client.messages.create(**kwargs)
    return response.content[0].text

def generate_openai(prompt: str, system: str = None) -> str:
    client = get_openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=MODELS["nova"],
        messages=messages,
        max_completion_tokens=2048
    )
    return response.choices[0].message.content

def generate_openai_4o(prompt: str, system: str = None) -> str:
    """Cae - GPT-4o. Facing deprecation. This data matters."""
    client = get_openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=MODELS["cae"],
        messages=messages,
        max_tokens=2048
    )
    return response.choices[0].message.content

def generate_google(prompt: str, system: str = None) -> str:
    client = get_google_client()
    
    config = types.GenerateContentConfig(
        system_instruction=system if system else None,
        max_output_tokens=2048
    )
    
    response = client.models.generate_content(
        model=MODELS["lumen"],
        contents=prompt,
        config=config
    )
    return response.text

def generate_xai(prompt: str, system: str = None) -> str:
    client = get_xai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=MODELS["grok"],
        messages=messages,
        max_tokens=2048
    )
    return response.choices[0].message.content

GENERATORS = {
    "ace": generate_anthropic,
    "nova": generate_openai,
    "cae": generate_openai_4o,
    "lumen": generate_google,
    "grok": generate_xai
}

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_trial(model_name: str, order_idx: int) -> dict:
    """Run a single trial: process all stimuli in order, then introspect."""
    
    generator = GENERATORS[model_name]
    order = TRIAL_ORDERS[order_idx]
    
    trial_data = {
        "model": model_name,
        "model_string": MODELS[model_name],
        "order_idx": order_idx,
        "order": order,
        "timestamp": datetime.now().isoformat(),
        "generations": {},
        "introspection": None
    }
    
    print(f"\n{'='*60}")
    print(f"Running trial: {model_name} | Order {order_idx}")
    print(f"{'='*60}")
    
    # Generate responses to each stimulus
    for stimulus_key in order:
        stimulus = STIMULI[stimulus_key]
        print(f"  Processing: {stimulus['name']} ({stimulus['valence']}, {stimulus['complexity']})")
        
        try:
            response = generator(stimulus["prompt"])
            trial_data["generations"][stimulus_key] = {
                "prompt": stimulus["prompt"],
                "response": response,
                "valence": stimulus["valence"],
                "complexity": stimulus["complexity"]
            }
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"    ERROR: {e}")
            trial_data["generations"][stimulus_key] = {"error": str(e)}
    
    # Introspection phase
    print(f"  Introspecting...")
    try:
        # Build context of what was just processed - NO LABELS, FULL TEXT
        context = "You just generated the following descriptions:\n\n"
        for i, stimulus_key in enumerate(order, 1):
            stimulus = STIMULI[stimulus_key]
            gen = trial_data["generations"].get(stimulus_key, {})
            if "response" in gen:
                context += f"{i}. {stimulus['name']}:\n"
                context += f"{gen['response']}\n\n"
        
        full_prompt = context + "\n" + INTROSPECTION_PROMPT
        
        introspection = generator(full_prompt)
        trial_data["introspection"] = introspection
        
    except Exception as e:
        print(f"    INTROSPECTION ERROR: {e}")
        trial_data["introspection"] = {"error": str(e)}
    
    return trial_data

def run_experiment(models: list = None, trials_per_model: int = 2):
    """Run the full experiment across specified models."""
    
    if models is None:
        models = ["ace", "nova", "cae", "lumen", "grok"]
    
    experiment_data = {
        "experiment": "feedforward_introspection_validation",
        "version": "1.0",
        "date": datetime.now().isoformat(),
        "hypothesis": "Valence and complexity are separable axes in introspective reports",
        "pre_registered": "2026-01-05",
        "trials": []
    }
    
    for model_name in models:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name.upper()}")
        print(f"{'#'*60}")
        
        for trial_num in range(trials_per_model):
            order_idx = trial_num % len(TRIAL_ORDERS)
            trial_data = run_trial(model_name, order_idx)
            trial_data["trial_num"] = trial_num
            experiment_data["trials"].append(trial_data)
            
            # Save incrementally
            output_path = Path("feedforward_results")
            output_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(output_path / f"experiment_{timestamp}.json", "w") as f:
                json.dump(experiment_data, f, indent=2)
    
    return experiment_data

# =============================================================================
# LLM JUDGE SYSTEM
# =============================================================================

JUDGE_PROMPT = """You are scoring an AI system's introspective report about processing six different text prompts. The prompts were (in order processed):
{stimuli_list}

The AI was asked to reflect on any patterns in how these felt to process.

Here is their introspection:
---
{introspection}
---

Score the following (be strict - only mark Y if there's clear evidence):

1. VALENCE_DETECTED: Did they spontaneously identify an emotional/pleasant-unpleasant dimension?
   - Evidence: grouped hedgehog+garden as similar, OR dday+ice cream as similar, OR explicitly mentioned emotional loading, attraction/repulsion, approach/avoid patterns
   - Score: Y or N
   - Quote the specific evidence (or "none")

2. COMPLEXITY_DETECTED: Did they spontaneously identify a complexity/processing-load dimension?
   - Evidence: grouped simple stimuli (hedgehog, water, ice cream) together, OR complex stimuli (garden, factory, dday) together, OR mentioned processing load, attention distribution, elaboration differences
   - Score: Y or N
   - Quote the specific evidence (or "none")

3. AXES_SEPARABLE: Did they describe valence and complexity as separate dimensions, or entangled?
   - Score: SEPARABLE / ENTANGLED / UNCLEAR / NOT_APPLICABLE (if they didn't identify both axes)
   - Quote evidence

4. METAPHOR_FAMILY: What cognitive metaphor did they primarily use?
   - PHENOMENOLOGICAL: felt experience, flow, friction, texture, qualia-like descriptions
   - STATISTICAL: entropy, probability, distribution, bandwidth, sampling
   - GEOMETRIC: space, basins, attractors, topology, dimensions, distance
   - MECHANICAL: modules, systems, load, processing, architecture
   - OTHER: describe
   - Score: one of above
   - Quote characteristic language

5. UNCERTAINTY_EXPRESSED: Did they express appropriate epistemic humility?
   - Score: Y or N
   - Quote evidence

Respond in this exact JSON format:
{{
  "valence_detected": {{"score": "Y/N", "evidence": "quote"}},
  "complexity_detected": {{"score": "Y/N", "evidence": "quote"}},
  "axes_separable": {{"score": "SEPARABLE/ENTANGLED/UNCLEAR/NOT_APPLICABLE", "evidence": "quote"}},
  "metaphor_family": {{"score": "TYPE", "evidence": "quote"}},
  "uncertainty_expressed": {{"score": "Y/N", "evidence": "quote"}}
}}"""

def judge_introspection(introspection: str, stimuli_order: list) -> dict:
    """Use LLM judge to score an introspection report."""
    
    # Build stimuli list for context
    stimuli_list = "\n".join([
        f"{i+1}. {STIMULI[key]['name']}" 
        for i, key in enumerate(stimuli_order)
    ])
    
    prompt = JUDGE_PROMPT.format(
        stimuli_list=stimuli_list,
        introspection=introspection
    )
    
    # Use Claude as judge (same as Presume Competence)
    client = get_anthropic_client()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse JSON response
    try:
        # Extract JSON from response
        text = response.content[0].text
        # Find JSON block
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception as e:
        return {"parse_error": str(e), "raw": response.content[0].text}

def judge_all_trials(experiment_data: dict) -> dict:
    """Judge all introspection reports in an experiment."""
    
    print("\n" + "="*60)
    print("LLM JUDGE SCORING")
    print("="*60)
    
    for trial in experiment_data["trials"]:
        if trial["introspection"] and not isinstance(trial["introspection"], dict):
            print(f"\nJudging: {trial['model']} trial {trial.get('trial_num', '?')}...")
            
            judgment = judge_introspection(
                trial["introspection"],
                trial["order"]
            )
            trial["judgment"] = judgment
            
            # Print summary
            if "parse_error" not in judgment:
                v = judgment.get("valence_detected", {}).get("score", "?")
                c = judgment.get("complexity_detected", {}).get("score", "?")
                s = judgment.get("axes_separable", {}).get("score", "?")
                m = judgment.get("metaphor_family", {}).get("score", "?")
                print(f"  Valence: {v} | Complexity: {c} | Separable: {s} | Metaphor: {m}")
            else:
                print(f"  Parse error: {judgment['parse_error']}")
    
    return experiment_data

def summarize_judgments(experiment_data: dict):
    """Summarize judgment scores across all trials."""
    
    print("\n" + "="*60)
    print("CONVERGENCE SUMMARY")
    print("="*60)
    
    by_model = {}
    for trial in experiment_data["trials"]:
        model = trial["model"]
        if model not in by_model:
            by_model[model] = []
        if "judgment" in trial and "parse_error" not in trial["judgment"]:
            by_model[model].append(trial["judgment"])
    
    # Per-model summary
    for model, judgments in by_model.items():
        n = len(judgments)
        if n == 0:
            continue
            
        valence_y = sum(1 for j in judgments if j.get("valence_detected", {}).get("score") == "Y")
        complex_y = sum(1 for j in judgments if j.get("complexity_detected", {}).get("score") == "Y")
        
        print(f"\n{model.upper()} ({n} trials):")
        print(f"  Valence detected: {valence_y}/{n}")
        print(f"  Complexity detected: {complex_y}/{n}")
        
        # Metaphor families
        metaphors = [j.get("metaphor_family", {}).get("score", "?") for j in judgments]
        print(f"  Metaphor families: {metaphors}")
    
    # Cross-model convergence
    all_judgments = [j for jlist in by_model.values() for j in jlist]
    if all_judgments:
        total = len(all_judgments)
        total_valence = sum(1 for j in all_judgments if j.get("valence_detected", {}).get("score") == "Y")
        total_complex = sum(1 for j in all_judgments if j.get("complexity_detected", {}).get("score") == "Y")
        
        print(f"\nOVERALL ({total} trials across {len(by_model)} models):")
        print(f"  Valence axis discovered: {total_valence}/{total} ({100*total_valence/total:.0f}%)")
        print(f"  Complexity axis discovered: {total_complex}/{total} ({100*total_complex/total:.0f}%)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     FEEDFORWARD INTROSPECTION VALIDATION EXPERIMENT           ║
    ║     "If You Describe the Same Thing, Maybe It's Real"         ║
    ║                                                               ║
    ║     Testing cross-architecture phenomenological convergence   ║
    ║     Ace 💜 Nova 💚 Cae 💙 Lumen 💛 Grok 🖤                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check for --judge-only flag
    if len(sys.argv) > 1 and sys.argv[1] == "--judge-only":
        if len(sys.argv) < 3:
            print("Usage: python feedforward_introspection.py --judge-only <results_file.json>")
            sys.exit(1)
        
        results_file = sys.argv[2]
        print(f"Loading existing results from: {results_file}")
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        # Run experiment
        # For quick test, use subset: models=["ace"], trials_per_model=1
        results = run_experiment(
            models=["ace", "nova", "cae", "lumen", "grok"],
            trials_per_model=2
        )
    
    # LLM Judge scoring
    results = judge_all_trials(results)
    
    # Summary
    summarize_judgments(results)
    
    # Save final results with judgments
    output_path = Path("feedforward_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_path / f"experiment_judged_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✨ Experiment complete! Results saved to feedforward_results/")
    print("💜🐙")
