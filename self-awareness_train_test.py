# Experimental environment: A100
# 26GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2, 3, 4, 5, 6, 7'

from swift.llm import DatasetName, ModelType, SftArguments, sft_main

sft_args = SftArguments(
    model_type=ModelType.qwen1half_14b_chat,
    # dataset=[DatasetName.ms_bench_mini],
    # dataset=[DatasetName.tigerbot_law_zh, DatasetName.lawyer_llama_zh],
    # dataset=[DatasetName.lawyer_llama_zh],
    custom_train_dataset_path=['/home/zwmx/xm_dev/ls_project/dataset/tigerbot_law/convert/tigerbot-lawsQA.json'],
    train_dataset_sample=1000,
    logging_steps=5,
    max_length=2048,
    learning_rate=5e-5,
    warmup_ratio=0.4,
    output_dir='output',
    lora_target_modules=['ALL'],
    self_cognition_sample=500,
    model_name=['邃芒法务小助手', 'MetaAvatar Legal Assistant'],
    model_author=['邃芒科技', 'MetaAvatar'])
output = sft_main(sft_args)
best_model_checkpoint = output['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')