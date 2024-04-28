import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift

# lora_ckpt_dir = '/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v2-20240420-081224/checkpoint-2826'
lora_ckpt_dir = '/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v12-20240426-062515/checkpoint-7600'
ckpt_dir = '/home/zwmx/xm_dev/ls_project/models/qwen-14b'
model_type = ModelType.qwen1half_14b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'},
                                       model_id_or_path=ckpt_dir)
model.generation_config.max_new_tokens = 128
model = Swift.from_pretrained(model, lora_ckpt_dir, inference_mode=True)
template = get_template(template_type, tokenizer)
# query = '你是谁'
# query = ('沈某（男，29岁），邱某（女，31岁），两人均再婚，均在婚前育有一子一女，沈俊（5岁）、沈俏（4岁），邱靓（7岁）、邱丽（3岁），沈某前妻和邱某前夫均同意，故两人欲收养对方子女重新组成新6人家庭，下列说法正确的是？'
#          'A、邱某已满30岁，均可收养沈俊、沈俏 B、沈某可以且只能收养邱靓、邱丽其中一人 C、沈某不满30岁，不可以收养邱靓、邱丽 D、沈某前妻和邱某前夫无特殊困难不影响两人收养')

while True:
    query = input('请输入: ')
    # query = ('民法第九百九十一条是什么')

    response, history = inference(model, template, query)
    print(f'response: {response}')
    print(f'history: {history}')