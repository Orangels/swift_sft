nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
swift sft \
    --model_type qwen1half-14b-chat \
    --model_id_or_path /yldm0226/models/Qwen1.5-14B-Chat \
    --model_revision master \
    --sft_type lora \
    --tuner_backend swift \
    --template_type qwen \
    --dtype AUTO \
    --output_dir /yldm0226/llm_sft_output \
    --ddp_backend nccl \
    --custom_train_dataset_path /yldm0226/data/1-Chinese_medical_dialogue_six_department.jsonl /yldm0226/data/2-HuatuoGPT2_sft_instruct_GPT4.jsonl /yldm0226/data/3-ChatMed_Consult-v0.3.jsonl /yldm0226/data/4-ChatMed_TCM-v0.2.jsonl /yldm0226/data/5-QiZhen_sft_20k.jsonl /yldm0226/data/6-Huatuo_Lite.jsonl /yldm0226/data/7-ZhongJing_CMtMedQA.jsonl /yldm0226/data/8-DISC-Med-SFT_released.jsonl /yldm0226/data/9-SZY_TCM_QA.jsonl \
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
    --save_total_limit 3 \
    --logging_steps 10 \
    --use_flash_attn false \
    --deepspeed default-zero3 \
    --save_only_model true
