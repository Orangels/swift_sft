# 直接使用app-ui
CUDA_VISIBLE_DEVICES=0,1,2,3 swift app-ui --ckpt_dir '/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v2-20240420-081224/checkpoint-2826-merged' --share True
# Merge LoRA增量权重并使用app-ui
# 如果你需要量化, 可以指定`--quant_bits 4`.
#CUDA_VISIBLE_DEVICES=0 swift export \
#    --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx' --merge_lora true max_model_len=40000
#CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx-merged'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift app-ui --ckpt_dir '/home/zwmx/xm_dev/ls_project/swift/scripts/ls/output/qwen1half-14b-chat/v8-20240427-133147/checkpoint-93' --merge_lora true --max_model_len=9000 --share True --server_name '0.0.0.0' --infer_backend vllm
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir '/home/zwmx/xm_dev/ls_project/swift/scripts/ls/output/qwen1half-14b-chat/v8-20240427-133147/checkpoint-93-merged' --model_type qwen1half-14b-chat --share True --max_model_len=9000 --server_name '0.0.0.0' --infer_backend vllm
CUDA_VISIBLE_DEVICES=0 swift app-ui --model_type qwen-7b-chat --infer_backend vllm --server_name '0.0.0.0'
CUDA_VISIBLE_DEVICES=0 swift app-ui --model_type qwen1half-14b-chat --infer_backend vllm --server_name '0.0.0.0' --max_model_len=9000