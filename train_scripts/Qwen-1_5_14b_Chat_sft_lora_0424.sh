#!/usr/bin/env bash
nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
swift sft \
    --model_type qwen1half-14b-chat \
    --model_id_or_path /home/zwmx/xm_dev/ls_project/models/qwen-14b \
    --model_revision master \
    --sft_type lora \
    --tuner_backend swift \
    --template_type qwen \
    --dtype AUTO \
    --output_dir /home/zwmx/xm_dev/ls_project/llm_sft_output \
    --ddp_backend nccl \
    --custom_train_dataset_path /home/zwmx/xm_dev/ls_project/dataset/DISC-Law-SFT_ls/train_data_law_DISC-Law-SFT-Pair.json /home/zwmx/xm_dev/ls_project/dataset/DISC-Law-SFT_ls/train_data_law.json /home/zwmx/xm_dev/ls_project/dataset/lawyer-llama/convert/lawyer-llamaQA-article.json /home/zwmx/xm_dev/ls_project/dataset/lawyer-llama/convert/lawyer-llamaQA.json /home/zwmx/xm_dev/ls_project/dataset/tigerbot_law/convert/tigerbot-lawsQA.json \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --use_flash_attn false \
    --save_only_model false \
    --self_cognition_sample 500 \
    --model_name 邃芒法务小助手 'MetaAvatar Legal Assistant' \
    --model_author 邃芒科技 MetaAvatar
