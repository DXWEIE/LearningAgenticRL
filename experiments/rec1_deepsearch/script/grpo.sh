
CUDA_VISIBLE_DEVICES=1,2,3 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model /data/coding/finetuned_model/lora_dft_final_0430/v0-20260430-212504/checkpoint-130-merged \
    --output_dir /data/coding/finetuned_model/grpo_final/ \
    --tuner_type lora \
    --reward_funcs deep_search_reward \
    --multi_turn_scheduler deep_search_scheduler \
    --max_turns 28 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 9122 \
    --vllm_server_pass_dataset true \
    --torch_dtype bfloat16 \
    --dataset /data/coding/web_shape_last_300_grpo_xml_final.jsonl \
    --overlong_filter true \
    --loss_scale default \
    --split_dataset_ratio 0 \
    --epsilon_high 0.28 \
    --max_completion_length 4096 \
    --max_length 26624 \
    --completion_length_limit_scope total \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --steps_per_generation 2 \
    --gradient_checkpointing true \
    --save_steps 30 \
    --overlong_filter true \
    --logging_steps 1 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --warmup_ratio 0.05 \
    --num_generations 6 \
    --temperature 0.8 \
    --deepspeed zero3_offload \
    --log_completions true \
    --log_entropy true \
    --num_iterations 1 \
    --rollout_importance_sampling_mode token_mask













