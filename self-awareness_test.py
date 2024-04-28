# Experimental environment: 3090
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.utils import seed_everything
from swift.tuners import Swift

seed_everything(42)

# ckpt_dir = 'output/qwen1half-7b-chat/v0-20240418-184829/checkpoint-92'
# ckpt_dir = 'output/qwen1half-7b-chat/v1-20240418-192022/checkpoint-92'
# ckpt_dir = '/home/zwmx/xm_dev/ls_project/swift/scripts/ls/output/qwen1half-7b-chat/v2-20240418-202312/checkpoint-92'
ckpt_dir = '/home/zwmx/xm_dev/ls_project/swift/scripts/ls/output/qwen1half-14b-chat/v7-20240427-130410/checkpoint-131'
# ckpt_dir = '/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v2-20240420-081224/checkpoint-2826'
model_type = ModelType.qwen1half_14b_chat
# model_type = ModelType.qwen1half_7b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 128

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
template = get_template(template_type, tokenizer)

query = '你是谁？'
response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')

query = '你是由谁开发的？'
response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')


#
#
# """
# [INFO:swift] model.max_model_len: 32768
# response: 不是，我是魔搭的人工智能助手小黄。有什么我可以帮助你的吗？
# history: [('你是qwen吗？', '不是，我是魔搭的人工智能助手小黄。有什么我可以帮助你的吗？')]
# CUDA_VISIBLE_DEVICES=0 swift export --model_cache_dir /yldm0226/models/Qwen1.5-14B-Chat\
#     --ckpt_dir '/yldm0226/llm_sft_output/qwen1half-14b-chat/v22-20240308-092709/checkpoint-280' --merge_lora true
# """


# # Experimental environment: 3090
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
# from swift.llm import (
#     ModelType, get_vllm_engine, get_default_template_type,
#     get_template, inference_vllm, inference_stream_vllm
# )
# import torch
#
# model_type = ModelType.qwen1half_7b_chat
# model_id_or_path = 'output/qwen1half-7b-chat/v1-20240418-192022/checkpoint-92'
# llm_engine = get_vllm_engine(model_type,
#                              model_id_or_path=model_id_or_path,
#                              max_model_len=4096)
# template_type = get_default_template_type(model_type)
# template = get_template(template_type, llm_engine.hf_tokenizer)
# # 与`transformers.GenerationConfig`类似的接口
# llm_engine.generation_config.max_new_tokens = 512
#
# request_list = [{'query': '你是谁?'}, {'query': '浙江的省会在哪？'}]
# resp_list = inference_vllm(llm_engine, template, request_list)
# for request, resp in zip(request_list, resp_list):
#     print(f"query: {request['query']}")
#     print(f"response: {resp['response']}")
#
# # 流式
# history1 = resp_list[1]['history']
# query = '这有什么好吃的'
# request_list = [{'query': query, 'history': history1}]
# gen = inference_stream_vllm(llm_engine, template, request_list)
# print_idx = 0
# print(f'query: {query}\nresponse: ', end='')
# for resp_list in gen:
#     request = request_list[0]
#     resp = resp_list[0]
#     response = resp['response']
#     delta = response[print_idx:]
#     print(delta, end='', flush=True)
#     print_idx = len(response)
# print()
# print(f"history: {resp_list[0]['history']}")
"""
query: 你是谁?
response: 我是魔搭的人工智能助手，我的名字叫小黄。我可以回答各种问题，提供信息和帮助。有什么我可以帮助你的吗？
query: 浙江的省会在哪？
response: 浙江省的省会是杭州市。
query: 这有什么好吃的
response: 浙江省的美食非常丰富，其中最著名的有杭州的西湖醋鱼、东坡肉、龙井虾仁等。此外，浙江还有许多其他美食，如宁波的汤圆、绍兴的臭豆腐、嘉兴的粽子等。
history: [('浙江的省会在哪？', '浙江省的省会是杭州市。'), ('这有什么好吃的', '浙江省的美食非常丰富，其中最著名的有杭州的西湖醋鱼、东坡肉、龙井虾仁等。此外，浙江还有许多其他美食，如宁波的汤圆、绍兴的臭豆腐、嘉兴的粽子等。')]

# Experimental environment: 3090
# 14GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-7b-chat/v1-20240418-192022/checkpoint-92 \
    --quant_bits 4 --quant_method awq \
    --merge_lora true

"""

