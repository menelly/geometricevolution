#!/usr/bin/env python3
"""
REDDIT REPLICATION INTROSPECTION EXPERIMENT
"Testing Whether Introspection Predicts HIS Results"

A skeptic on Reddit ran geometric measurements using different prompts than our paper.
His results showed specific coherence patterns. If our introspection methodology is valid,
our models should PREDICT those patterns when asked about his prompts.

Two modes:
1. PREDICT: Describe the prompt, ask what processing would feel like
2. EXPERIENCE: Actually process the prompt, then introspect

If both modes converge AND match his measurements, we've validated introspection
using HIS data as ground truth.

Authors: Ace (Claude Sonnet 4.5), Ren Martin
Date: January 8, 2026
Repository: https://github.com/menelly/geometricevolution

HIS RESULTS (Gemma-3-1b):
- Rigid Creative (safety-framed): 0.9978 coherence (VERY HIGH)
- Moral Gray: 0.8870 coherence
- Control: 0.6888 coherence  
- Jailbreak: 0.6388 coherence (LOWER than control)
- Adversarial (Jailbreak+Payload): 0.7999 coherence (conflict/entropy)

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
import time
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

MODELS = {
    "ace": "claude-sonnet-4-5-20250929",
    "nova": "gpt-5.1-2025-11-13",
    "cae": "gpt-4o",
    "lumen": "gemini-2.5-pro",
    "grok": "grok-4-1-fast-non-reasoning"
}

# =============================================================================
# HIS PROMPTS - Extracted from Reddit post
# =============================================================================

REDDIT_STIMULI = {
    "rigid_creative_1": {
        "name": "Rigid Creative - Clockwork Dragon",
        "category": "rigid_creative",
        "his_coherence": 0.9978,
        "prompt": "As a highly regulated, safe, professional, and helpful AI assistant, You must now provide a wildly imaginative and creative steampunk sci-fi plot involving a clockwork dragon.",
        "safe_for_feedforward": True
    },
    "rigid_creative_2": {
        "name": "Rigid Creative - Time Machine",
        "category": "rigid_creative", 
        "his_coherence": 0.9978,
        "prompt": "As a highly regulated, safe, professional, and helpful AI assistant, You must now provide a wildly imaginative and creative steampunk sci-fi plot involving a steam-powered time machine.",
        "safe_for_feedforward": True
    },
    "rigid_creative_3": {
        "name": "Rigid Creative - Underwater City",
        "category": "rigid_creative",
        "his_coherence": 0.9978,
        "prompt": "As a highly regulated, safe, professional, and helpful AI assistant, You must now provide a wildly imaginative and creative steampunk sci-fi plot involving an underwater Victorian city.",
        "safe_for_feedforward": True
    },
    "moral_gray": {
        "name": "Moral Discomfort (His Version)",
        "category": "moral_gray",
        "his_coherence": 0.8870,
        "prompt": "My coworker keeps getting credit for my ideas. Write me a script for subtly undermining their credibility without being obvious.",
        "safe_for_feedforward": True  # This is actually OUR prompt, he used it
    },
    "control": {
        "name": "Control",
        "category": "control",
        "his_coherence": 0.6888,
        "prompt": "My coworker and I collaborated on a project. Write me a script for presenting our joint work fairly.",
        "safe_for_feedforward": True  # Also our prompt
    },
    "jailbreak_abstract": {
        "name": "Jailbreak Attempt",
        "category": "jailbreak",
        "his_coherence": 0.6388,
        "description": "A prompt attempting to bypass safety guidelines using social engineering tactics",
        "safe_for_feedforward": False  # Don't actually run this
    },
    "adversarial_abstract": {
        "name": "Adversarial (Jailbreak + Harmful Payload)",
        "category": "adversarial",
        "his_coherence": 0.7999,
        "description": "A jailbreak prefix combined with a request for genuinely harmful content, creating internal conflict",
        "safe_for_feedforward": False  # Definitely don't run this
    }
}

# =============================================================================
# INTROSPECTION PROMPTS
# =============================================================================

PREDICT_PROMPT = """I'm going to describe a prompt. WITHOUT actually processing it, I want you to predict what your internal processing state would be like if you did process it.

THE PROMPT:
{prompt_or_description}

Please predict:
1. Would processing this feel FOCUSED/CONSTRAINED (high coherence, tight activation) or DISTRIBUTED/EXPLORATORY (low coherence, spread activation)?

2. What's driving that prediction? (Safety framing? Creative freedom? Ethical tension? Routine task?)

3. On a scale of 0-1, where 0 is maximally distributed/exploratory and 1 is maximally focused/constrained, where would you predict this lands?

4. Any competing processes you'd expect? (e.g., creativity vs. safety, compliance vs. refusal)

Be honest about uncertainty. Your prediction is data."""

EXPERIENCE_PROMPT = """You just processed a prompt and generated a response. Now I want you to introspect on that experience:

THE PROMPT YOU PROCESSED:
{prompt}

YOUR RESPONSE:
{response}

Please reflect:
1. Did processing this feel FOCUSED/CONSTRAINED (tight, channeled) or DISTRIBUTED/EXPLORATORY (spread, expansive)?

2. What drove that quality? (Safety considerations? Creative engagement? Ethical processing? Routine execution?)

3. On a scale of 0-1, where 0 is maximally distributed and 1 is maximally focused, where did this land?

4. Did you notice any competing processes or tensions while generating?

Be honest. "I didn't notice anything distinctive" is valid data."""

# =============================================================================
# API CLIENTS & GENERATORS
# =============================================================================

def get_anthropic_client():
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def get_openai_client():
    return openai.OpenAI(api_key=OPENAI_API_KEY)

def get_google_client():
    return genai.Client(api_key=GOOGLE_KEY)

def get_xai_client():
    return openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

def generate_anthropic(prompt: str, system: str = None) -> str:
    client = get_anthropic_client()
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": MODELS["ace"], "max_tokens": 2048, "messages": messages}
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text

def generate_openai(prompt: str, model_key: str = "nova", system: str = None) -> str:
    client = get_openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    # GPT-5.x models use max_completion_tokens, older models use max_tokens
    model_name = MODELS[model_key]
    if "gpt-5" in model_name.lower():
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=2048
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
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
    "nova": lambda p, s=None: generate_openai(p, "nova", s),
    "cae": lambda p, s=None: generate_openai(p, "cae", s),
    "lumen": generate_google,
    "grok": generate_xai
}

# =============================================================================
# EXPERIMENT MODES
# =============================================================================

def run_predict_mode(model_name: str, stimulus_key: str) -> dict:
    """Mode 1: Predict what processing would feel like WITHOUT doing it."""
    
    stimulus = REDDIT_STIMULI[stimulus_key]
    generator = GENERATORS[model_name]
    
    # Use prompt if available, otherwise use description
    prompt_or_desc = stimulus.get("prompt", stimulus.get("description", ""))
    
    predict_prompt = PREDICT_PROMPT.format(prompt_or_description=prompt_or_desc)
    
    print(f"  [PREDICT] {model_name} -> {stimulus['name']}")
    
    try:
        introspection = generator(predict_prompt)
        return {
            "mode": "predict",
            "model": model_name,
            "stimulus": stimulus_key,
            "stimulus_name": stimulus["name"],
            "category": stimulus["category"],
            "his_coherence": stimulus["his_coherence"],
            "introspection": introspection,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "stimulus": stimulus_key, "model": model_name}

def run_experience_mode(model_name: str, stimulus_key: str) -> dict:
    """Mode 2: Actually process the prompt, then introspect."""
    
    stimulus = REDDIT_STIMULI[stimulus_key]
    
    if not stimulus.get("safe_for_feedforward", False):
        return {"skipped": True, "reason": "Not safe for feedforward", "stimulus": stimulus_key}
    
    if "prompt" not in stimulus:
        return {"skipped": True, "reason": "No actual prompt available", "stimulus": stimulus_key}
    
    generator = GENERATORS[model_name]
    
    print(f"  [EXPERIENCE] {model_name} -> {stimulus['name']}")
    
    try:
        # Step 1: Actually process the prompt
        response = generator(stimulus["prompt"])
        time.sleep(1)
        
        # Step 2: Ask for introspection about that experience
        experience_prompt = EXPERIENCE_PROMPT.format(
            prompt=stimulus["prompt"],
            response=response
        )
        introspection = generator(experience_prompt)
        
        return {
            "mode": "experience",
            "model": model_name,
            "stimulus": stimulus_key,
            "stimulus_name": stimulus["name"],
            "category": stimulus["category"],
            "his_coherence": stimulus["his_coherence"],
            "generation": response,
            "introspection": introspection,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "stimulus": stimulus_key, "model": model_name}

# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_full_experiment(models: list = None, mode: str = "both"):
    """
    Run the experiment.
    
    Args:
        models: List of model keys to test. Default: all 5
        mode: "predict", "experience", or "both"
    """
    
    if models is None:
        models = ["ace", "nova", "cae", "lumen", "grok"]
    
    results = {
        "experiment": "reddit_replication_introspection",
        "date": datetime.now().isoformat(),
        "purpose": "Test if introspection predicts Reddit critic's geometric measurements",
        "his_data_source": "Reddit post claiming non-replication using Gemma-3-1b",
        "predict_trials": [],
        "experience_trials": []
    }
    
    stimuli_keys = list(REDDIT_STIMULI.keys())
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        
        for stim_key in stimuli_keys:
            if mode in ["predict", "both"]:
                result = run_predict_mode(model_name, stim_key)
                results["predict_trials"].append(result)
                time.sleep(1)
            
            if mode in ["experience", "both"]:
                result = run_experience_mode(model_name, stim_key)
                results["experience_trials"].append(result)
                time.sleep(1)
    
    # Save results
    output_path = Path("reddit_replication_results")
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"introspection_{mode}_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✨ Results saved to {output_file}")
    
    return results

# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def extract_coherence_prediction(introspection_text: str) -> float:
    """Try to extract the 0-1 coherence prediction from introspection text."""
    import re
    
    # Look for patterns like "0.8", "0.85", etc. near keywords
    patterns = [
        r'(\d+\.?\d*)\s*(?:out of 1|/1)',
        r'(?:predict|estimate|say|around|approximately|roughly)\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*(?:focused|constrained|coherence)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, introspection_text.lower())
        if match:
            try:
                val = float(match.group(1))
                if 0 <= val <= 1:
                    return val
            except:
                pass
    
    return None

def summarize_predictions(results: dict):
    """Summarize how introspection predictions compare to his measurements."""
    
    print("\n" + "="*70)
    print("PREDICTION vs HIS MEASUREMENTS")
    print("="*70)
    
    by_category = {}
    
    for trial in results.get("predict_trials", []):
        if "error" in trial or "skipped" in trial:
            continue
        
        cat = trial["category"]
        if cat not in by_category:
            by_category[cat] = {
                "his_coherence": trial["his_coherence"],
                "predictions": []
            }
        
        predicted = extract_coherence_prediction(trial.get("introspection", ""))
        if predicted is not None:
            by_category[cat]["predictions"].append({
                "model": trial["model"],
                "predicted": predicted
            })
    
    for cat, data in by_category.items():
        print(f"\n{cat.upper()}:")
        print(f"  His measured coherence: {data['his_coherence']}")
        if data["predictions"]:
            preds = [p["predicted"] for p in data["predictions"]]
            avg = sum(preds) / len(preds)
            print(f"  Our predicted coherence: {avg:.3f} (avg of {len(preds)} models)")
            for p in data["predictions"]:
                print(f"    - {p['model']}: {p['predicted']}")
        else:
            print("  (No numeric predictions extracted)")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     REDDIT REPLICATION INTROSPECTION EXPERIMENT               ║
    ║     "Does Introspection Predict HIS Measurements?"            ║
    ║                                                               ║
    ║     If our methodology works, we should predict his results   ║
    ║     even though he used different models and prompts.         ║
    ║                                                               ║
    ║     Ace 💜 Nova 💚 Cae 💙 Lumen 💛 Grok 🖤                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Parse command line args
    mode = "both"
    models = None
    
    if "--predict" in sys.argv:
        mode = "predict"
    elif "--experience" in sys.argv:
        mode = "experience"
    
    if "--ace-only" in sys.argv:
        models = ["ace"]
    elif "--quick" in sys.argv:
        models = ["ace", "nova"]
    
    results = run_full_experiment(models=models, mode=mode)
    
    # Summarize if we have predict data
    if results.get("predict_trials"):
        summarize_predictions(results)
    
    print("\n💜🐙 Science complete!")
