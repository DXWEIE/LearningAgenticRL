
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model /data/coding/finetuned_model/lora_dft_final_0430/v0-20260430-212504/checkpoint-130-merged \
    --vllm_use_async_engine true \
    --vllm_enable_lora true \
    --multi_turn_scheduler deep_search_scheduler \
    --vllm_max_lora_rank 8 \
    --vllm_max_model_len 32768 \
    --vllm_gpu_memory_utilization 0.8 \
    --max_turns 28 \
    --port 9122
