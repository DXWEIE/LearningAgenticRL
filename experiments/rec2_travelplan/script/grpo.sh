
CUDA_VISIBLE_DEVICES=1,2,3 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model /data/coding/travel_finetune/dft_0517/v0-20260517-202958/checkpoint-55-merged \
    --output_dir /data/coding/travel_finetune/grpo_final/ \
    --tuner_type lora \
    --reward_funcs open_traval_ori_reward \
    --multi_turn_scheduler open_traval_scheduler \
    --max_turns 28 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 9124 \
    --vllm_server_pass_dataset true \
    --torch_dtype bfloat16 \
    --dataset /data/coding/trip_plan_rl_dataset.jsonl \
    --overlong_filter true \
    --loss_scale default \
    --split_dataset_ratio 0 \
    --epsilon_high 0.28 \
    --max_completion_length 8192 \
    --max_length 28672 \
    --completion_length_limit_scope total \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --steps_per_generation 4 \
    --gradient_checkpointing true \
    --save_steps 5 \
    --overlong_filter true \
    --logging_steps 1 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --warmup_ratio 0.05 \
    --num_generations 4 \
    --temperature 0.8 \
    --deepspeed zero3_offload \
    --log_completions true \
    --log_entropy true \
    --num_iterations 1 \
    --rollout_importance_sampling_mode token_mask













