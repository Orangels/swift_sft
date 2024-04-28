# merge LoRA增量权重并部署
# 如果你需要量化, 可以指定`--quant_bits 4`.
#CUDA_VISIBLE_DEVICES=0 swift export \
#    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true
CUDA_VISIBLE_DEVICES=1 swift deploy --ckpt_dir '/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v2-20240420-081224/checkpoint-2826-merged'