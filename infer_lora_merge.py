import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)
ckpt_dir = '/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v7-20240423-125134/checkpoint-2830-merged'
model_type = ModelType.qwen1half_14b_chat
#llm_engine = get_vllm_engine(model_type)
llm_engine = get_vllm_engine(model_type, model_id_or_path=ckpt_dir, max_model_len=9000)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 256

request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")

history1 = resp_list[1]['history']
request_list = [{'query': '这有什么好吃的', 'history': history1}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
    print(f"history: {resp['history']}")

"""Out[0]
query: 你好!
response: 你好！很高兴为你服务。有什么我可以帮助你的吗？
query: 浙江的省会在哪？
response: 浙江省会是杭州市。
query: 这有什么好吃的
response: 杭州是一个美食之城，拥有许多著名的菜肴和小吃，例如西湖醋鱼、东坡肉、叫化童子鸡等。此外，杭州还有许多小吃店，可以品尝到各种各样的本地美食。
history: [('浙江的省会在哪？', '浙江省会是杭州市。'), ('这有什么好吃的', '杭州是一个美食之城，拥有许多著名的菜肴和小吃，例如西湖醋鱼、东坡肉、叫化童子鸡等。此外，杭州还有许多小吃店，可以品尝到各种各样的本地美食。')]
"""
