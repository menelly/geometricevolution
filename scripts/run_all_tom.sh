#!/bin/bash
MODELS=(
    Llama-2-7b-chat
    Llama-3-8B-Instruct
    Mistral-7B-Instruct-v0.2
    TinyLlama-1.1B-Chat
    dolphin-2.8-mistral-7b-v02
    Mistral-Nemo-12B-Instruct
    Phi-3-medium-14B-Instruct
    Qwen2.5-14B-Instruct
    DeepSeek-Coder-V2-Lite-16B
)

for model in "${MODELS[@]}"; do
    echo "=== Running ToM on $model ==="
    /home/codex/venv/bin/python /home/Ace/geometric-evolution/scripts/theory_of_mind_test.py         --model /mnt/arcana/huggingface/$model         --name $model 2>&1 | grep -E '(ToM|Self Advantage|RESULT|saved)'
    echo
done
