import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)
from swift.tuners import Swift

ckpt_dir = '/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v2-20240420-081224/checkpoint-2826-merged'
model_type = ModelType.qwen1half_14b_chat
template_type = get_default_template_type(model_type)

llm_engine = get_vllm_engine(model_type, model_id_or_path=ckpt_dir)
tokenizer = llm_engine.hf_tokenizer
template = get_template(template_type, tokenizer)
query = '你好'
resp = inference_vllm(llm_engine, template, [{'query': query}])[0]
print(f"response: {resp['response']}")
print(f"history: {resp['history']}")