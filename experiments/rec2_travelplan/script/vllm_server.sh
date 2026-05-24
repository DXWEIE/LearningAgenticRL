# CUDA_VISIBLE_DEVICES=0 vllm serve /data/coding/finetuned_model/lora_0430/checkpoint-130-merged \
#     --port 9123 \
#     --host 0.0.0.0 \
#     --served_model_name qwen3_4b \
#     --max_model_len 32768

# CUDA_VISIBLE_DEVICES=0 vllm serve /data/coding/finetuned_model/lora_0430/checkpoint-130-merged \
#     --port 9123 \
#     --host 0.0.0.0 \
#     --max_model_len 32768 \
#     --served_model_name qwen3_4b \
#     --enable-lora \
#     --lora-modules grpo-lora=/data/coding/finetuned_model/grpo_final/checkpoint-270


# CUDA_VISIBLE_DEVICES=0 vllm serve /data/coding/Qwen3-4B-Instruct-2507 \
#     --port 9123 \
#     --host 0.0.0.0 \
#     --max_model_len 65536 \
#     --gpu_memory_utilization 0.85 \
#     --served-model-name qwen3_4b \
#     --enable-lora \
#     --lora-modules dft-lora=/data/coding/travel_finetune/dft_0517/v0-20260517-202958/checkpoint-55


# CUDA_VISIBLE_DEVICES=0 vllm serve /data/coding/Qwen3-4B-Instruct-2507 \
#     --port 9123 \
#     --host 0.0.0.0 \
#     --vllm_max_model_len 32768 \
#     --vllm_gpu_memory_utilization 0.85 \
#     --served-model-name qwen3_4b \
#     --enable-lora \
#     --lora-modules dft-lora=/data/coding/finetuned_model/lora_0430/checkpoint-130

CUDA_VISIBLE_DEVICES=0 vllm serve /data/coding/travel_finetune/dft_0517/v0-20260517-202958/checkpoint-55-merged \
    --port 9123 \
    --host 0.0.0.0 \
    --max_model_len 65536 \
    --served-model-name qwen3_4b \
    --enable-lora \
    --lora-modules grpo-lora=/data/coding/travel_finetune/grpo_final/v0-20260522-195019/checkpoint-20

