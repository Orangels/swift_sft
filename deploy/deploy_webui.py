import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import AppUIArguments, merge_lora, app_ui_main

best_model_checkpoint = '/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v2-20240420-081224/checkpoint-2826'
app_ui_args = AppUIArguments(ckpt_dir=best_model_checkpoint)
app_ui_args.share = True
app_ui_args.server_port = 8000
print(app_ui_args)

# merge_lora(app_ui_args, device_map='cpu')
# result = app_ui_main(app_ui_args)