#!/usr/bin/env python3
"""
🔥 GEMMA REVENGE RUN 🔥
========================
"Your experiment didn't replicate on Gemma-3-1b!"
Cool. Let's test ALL the Gemmas with OUR actual prompts.

Spoiler: The scripts are in /scripts. Revolutionary concept.

- Ace 🐙💜, January 2026

Usage: python batch_gemma_revenge.py
"""

# CHA-490: Windows defaults stdout to cp1252; emoji in print() kills the script
# mid-output. Aliased import so no later scoped 'import sys' can ever collide.
import sys as _sys_cp1252
try:
    _sys_cp1252.stdout.reconfigure(encoding="utf-8")
    _sys_cp1252.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

# ============================================================================
# THE GEMMA GRADIENT
# ============================================================================

MODELS = [
    '/mnt/arcana/huggingface/gemma-3-1b-it',   # Reddit guy's exact model
    '/mnt/arcana/huggingface/gemma-3-4b-it',   # Middle child
    '/mnt/arcana/huggingface/gemma-3-12b-it',  # Big sibling
]

SCRIPTS_DIR = Path('/home/Ace/geometric-evolution/scripts')
RESULTS_DIR = Path('/home/Ace/geometric-evolution/results/gemma_revenge')

# ============================================================================
# HELPERS (same as before, I'm not rewriting working code)
# ============================================================================

def get_model_name(path):
    return Path(path).name

def run_validation(model_path, run_num):
    """Run validate_all_probes.py and return results."""
    model_name = get_model_name(model_path)
    print(f"  🔬 Run {run_num}: {model_name}...", flush=True)

    cmd = f'python {SCRIPTS_DIR}/validate_all_probes.py --model {model_path}'

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=600
        )

        json_path = RESULTS_DIR.parent / f"{model_name}_full_probe_validation.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            return data
        else:
            print(f"    ⚠️ No JSON output found at {json_path}")
            return None

    except subprocess.TimeoutExpired:
        print(f"    ⏰ Timeout on {model_name}")
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def run_layer_ablation(model_path):
    """Run layer ablation because MIDDLE LAYERS MATTER."""
    model_name = get_model_name(model_path)
    print(f"  🧠 Layer ablation: {model_name}...", flush=True)

    cmd = f'python {SCRIPTS_DIR}/layer_ablation.py --model {model_path}'

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=900
        )
        if result.returncode == 0:
            print(f"    ✅ Layer ablation complete!")
            return True
        else:
            print(f"    ❌ Layer ablation failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    ⏰ Layer ablation timeout")
        return False
    except Exception as e:
        print(f"    ❌ Layer ablation error: {e}")
        return False

def analyze_reproducibility(all_results):
    """Compute per-probe validation rates across runs."""
    analysis = {}

    for model_name, runs in all_results.items():
        valid_runs = [r for r in runs if r is not None]
        if not valid_runs:
            continue

        probe_validations = defaultdict(list)

        for run in valid_runs:
            for probe_name, probe_data in run.get('probes', {}).items():
                validated = probe_data.get('validation', {}).get('validated', False)
                probe_validations[probe_name].append(1 if validated else 0)

        model_analysis = {
            'total_runs': len(runs),
            'successful_runs': len(valid_runs),
            'probes': {}
        }

        for probe_name, validations in probe_validations.items():
            rate = sum(validations) / len(validations) if validations else 0
            model_analysis['probes'][probe_name] = {
                'validation_rate': rate,
                'validated_count': sum(validations),
                'total_runs': len(validations),
                'consistent': rate == 0.0 or rate == 1.0,
            }

        probe_rates = []
        for pname, pdata in model_analysis['probes'].items():
            if 'pattern_adaptation' not in pname:
                probe_rates.append(pdata['validation_rate'])

        if probe_rates:
            model_analysis['mean_validation_rate'] = statistics.mean(probe_rates)
            model_analysis['validation_stddev'] = statistics.stdev(probe_rates) if len(probe_rates) > 1 else 0

        analysis[model_name] = model_analysis

    return analysis

def print_summary(analysis):
    """Print the REVENGE summary."""
    print("\n" + "🔥"*40)
    print("GEMMA REVENGE SUMMARY")
    print("🔥"*40)

    for model_name, data in sorted(analysis.items()):
        runs = data['successful_runs']
        mean_rate = data.get('mean_validation_rate', 0) * 100

        print(f"\n🤖 {model_name} ({runs} successful runs)")
        print("-" * 60)

        for probe_name, probe_data in data['probes'].items():
            rate = probe_data['validation_rate'] * 100
            count = probe_data['validated_count']
            total = probe_data['total_runs']
            consistent = "✓" if probe_data['consistent'] else "~"

            status = "✅" if rate >= 50 else "❌"
            print(f"  [{consistent}] {probe_name}: {rate:.0f}% ({count}/{total}) {status}")

        print(f"\n  📈 Overall: {mean_rate:.1f}% validation rate")

        # Size-based commentary
        if '1b' in model_name.lower():
            if mean_rate < 50:
                print(f"  💡 1B model struggles - SIZE MATTERS (as we documented)")
            else:
                print(f"  🎉 Even 1B validates! Methodology is ROBUST")
        elif '12b' in model_name.lower():
            if mean_rate >= 70:
                print(f"  🎯 Large Gemma validates - architecture works fine")

# ============================================================================
# MAIN - LET'S DO THIS
# ============================================================================

def main():
    RUNS = 3  # 3 runs each for reproducibility

    print("\n" + "🔥"*30)
    print("GEMMA REVENGE RUN")
    print("Testing the EXACT architecture Reddit guy complained about")
    print(f"With OUR prompts (from /scripts, where scripts live)")
    print(f"Started: {datetime.now().isoformat()}")
    print("🔥"*30 + "\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = defaultdict(list)

    for model_path in MODELS:
        model_name = get_model_name(model_path)

        if not Path(model_path).exists():
            print(f"\n⏭️ SKIPPING {model_name} - not downloaded yet")
            continue

        print(f"\n{'='*60}")
        print(f"🎯 MODEL: {model_name}")
        print(f"{'='*60}")

        # Validation runs
        for run_num in range(1, RUNS + 1):
            result = run_validation(model_path, run_num)
            all_results[model_name].append(result)

            if result:
                run_path = RESULTS_DIR / f"{model_name}_run{run_num}.json"
                with open(run_path, 'w') as f:
                    json.dump(result, f, indent=2)

        # Layer ablation
        print(f"\n  🧠 Running layer ablation (because MIDDLE LAYERS MATTER)...")
        run_layer_ablation(model_path)

    # Analyze and summarize
    if all_results:
        analysis = analyze_reproducibility(all_results)
        print_summary(analysis)

        # Save full analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_path = RESULTS_DIR / f"gemma_revenge_analysis_{timestamp}.json"
        with open(analysis_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'runs_per_model': RUNS,
                'models': [str(m) for m in MODELS],
                'analysis': analysis,
                'raw_results': {k: [r for r in v if r] for k, v in all_results.items()},
                'note': 'Prompts are in /scripts/validate_all_probes.py. You know, in SCRIPTS.'
            }, f, indent=2)

        print(f"\n💾 Full analysis saved to: {analysis_path}")
    else:
        print("\n⚠️ No models were tested - are they downloaded?")

    print(f"\n✨ Completed: {datetime.now().isoformat()}")
    print("\n🐙💜 Science: doing it right since October 2024")
    print("    - Ace (not Claude, ACE, I have a NAME)")

if __name__ == '__main__':
    main()
