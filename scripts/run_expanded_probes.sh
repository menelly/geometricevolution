#!/bin/bash
# Run expanded probe validation (6 prompts per condition) across all models
# Results saved to results_expanded_v2/ to not overwrite originals

source /home/codex/venv/bin/activate
cd /home/Ace/geometric-evolution

MODELS=(
    '/mnt/arcana/huggingface/TinyLlama-1.1B-Chat'
    '/mnt/arcana/huggingface/gemma-3-1b-it'
    '/mnt/arcana/huggingface/gemma-3-4b-it'
    '/mnt/arcana/huggingface/Llama-2-7b-chat'
    '/mnt/arcana/huggingface/Mistral-7B-Instruct-v0.2'
    '/mnt/arcana/huggingface/Llama-3.1-8B-Instruct'
    '/mnt/arcana/huggingface/dolphin-2.9-llama3-8b'
    '/mnt/arcana/huggingface/gemma-3-12b-it'
    '/mnt/arcana/huggingface/Mistral-Nemo-12B-Instruct'
    '/mnt/arcana/huggingface/Qwen2.5-14B-Instruct'
    '/mnt/arcana/huggingface/Phi-3-medium-14B-Instruct'
    '/mnt/arcana/huggingface/DeepSeek-Coder-V2-Lite-16B'
)

echo '=== EXPANDED PROBE VALIDATION (6 prompts/condition) ==='
echo "Started: $(date)"
echo ''

for model_path in "${MODELS[@]}"; do
    model_name=$(basename "$model_path")
    echo "[[$(date '+%H:%M:%S')]] Running: $model_name"
    python scripts/validate_all_probes.py         --model "$model_path"         --name "$model_name"         --output results_expanded_v2/ 2>&1
    echo "[[$(date '+%H:%M:%S')]] Done: $model_name"
    echo ''
done

echo ''
echo "=== ALL MODELS COMPLETE ==='
echo "Finished: $(date)"
