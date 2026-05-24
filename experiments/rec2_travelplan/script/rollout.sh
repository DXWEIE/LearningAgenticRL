
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model /data/coding/travel_finetune/dft_0517/v0-20260517-202958/checkpoint-55-merged \
    --vllm_use_async_engine true \
    --vllm_enable_lora true \
    --multi_turn_scheduler open_traval_scheduler \
    --vllm_max_lora_rank 8 \
    --vllm_max_model_len 32768 \
    --vllm_gpu_memory_utilization 0.8 \
    --max_turns 28 \
    --port 9123
