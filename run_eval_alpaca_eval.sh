

export DEBUG=1

reference_models="meta-llama/Llama-3.3-70B-Instruct-Turbo,Qwen/Qwen2.5-72B-Instruct-Turbo,meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo,meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

python generate_for_alpaca_eval.py \
    --model="Qwen/Qwen2.5-72B-Instruct-Turbo" \
    --output-path="outputs/Qwen-72B-round-1_MoA-Lite.json" \
    --reference-models=${reference_models} \
    --rounds 1 \
    --num-proc 12

alpaca_eval --model_outputs outputs/Qwen-72B-round-1_MoA-Lite.json --reference_outputs alpaca_eval/results/gpt4_1106_preview/model_outputs.json --output_path leaderboard

