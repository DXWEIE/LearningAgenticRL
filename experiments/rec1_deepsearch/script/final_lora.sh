
nproc_per_node=4
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
--model /data/coding/Qwen3-4B-Instruct-2507 \
--dataset /data/coding/df_final_training_RFT_0430.jsonl \
--tuner_type lora \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--output_dir /data/coding/finetuned_model/lora_dft_final_0430/ \
--enable_dft_loss true \
--num_train_epochs 5 \
--lorap_lr_ratio 10 \
--save_steps 20 \
--eval_steps 20 \
--save_total_limit 2 \
--logging_steps 10 \
--seed 42 \
--max_length 25600 \
--learning_rate 1e-4 \
--init_weights true \
--lora_rank 8 \
--lora_alpha 32 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--adam_epsilon 1e-08 \
--weight_decay 0.1 \
--gradient_accumulation_steps 4 \
--max_grad_norm 1 \
--lr_scheduler_type cosine \
--warmup_ratio 0.05 \
--warmup_steps 0 \
--gradient_checkpointing true \
--deepspeed zero3