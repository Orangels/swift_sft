# Experimental environment: 4 * A100
# 4 * 70GB GPU memory
nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model_type qwen1half-72b-chat \
    --dataset ms-bench-mini \
    --train_dataset_sample 1000 \
    --logging_steps 5 \
    --max_length 4096 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --output_dir output \
    --lora_target_modules ALL \
    --self_cognition_sample 500 \
    --model_name 邃芒法务小助手 'MetaAvatar Legal Assistant' \
    --model_author 邃芒科技 MetaAvatar \
    --deepspeed default-zero3