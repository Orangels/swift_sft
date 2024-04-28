import json
import os
import re
import sqlite3
from cn2an import cn2an, an2cn

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)

import gradio as gr

ckpt_dir = '/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v14-20240427-075629/checkpoint-2346-merged'
# ckpt_dir = '/home/zwmx/xm_dev/ls_project/models/qwen-14b'
model_type = ModelType.qwen1half_14b_chat
# llm_engine = get_vllm_engine(model_type)
llm_engine = get_vllm_engine(model_type, model_id_or_path=ckpt_dir,
                             max_model_len=9000)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 2048

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
conn = sqlite3.connect('../sqlite/law.db')
cursor = conn.cursor()





while True:
    # 说一下民法典中行政法的第三条是什么法
    #说一下经济法, 中华人民共和国海域使用管理法第三条是什么法
    #说一下经济法第三条是什么法
    query = input("请输入: ")
    pattern_int = r'(\d+条)'
    pattern = r"([一二三四五六七八九十百千万零壹贰叁肆伍陆柒捌玖拾佰仟萬]+条)"

    match_int_list = list(re.finditer(pattern_int, query))
    if match_int_list:
        for i in range(len(match_int_list)):
            match_int = match_int_list[i]
            start_index = match_int.start()
            end_index = match_int.end()
            an_number = match_int.group()
            print(an_number)
            number = an2cn(an_number[:-1])
            query = query[:start_index] + number + query[end_index - 1:]
            print(query)


    match = re.search(pattern, query)

    result = ''
    str_front = ''
    str_behand = ''
    law_str = ''
    #0 front, 1 behand
    query_str_mode = 0

    if match:
        start_index = match.start()
        end_index = match.end()
        number = match.group()
        print(number)
        # str_front = query[:start_index - 1]
        str_front = query[:start_index]
        if end_index < len(query):
            str_behand = query[end_index:]
        print(f'str_front -- {str_front}')
        print(f'str_behand -- {str_behand}')

        pattern_law_str = r"(法)"
        # match_law_str = re.search(pattern_law_str, query)
        match_law_str_list = list(re.finditer(pattern_law_str, str_front))

        if match_law_str_list:
            match_law_str = match_law_str_list[-1]
            start_index = match_law_str.start()
            end_index = match_law_str.end()
            law_char = match_law_str.group()
            law_str = str_front[start_index-1:start_index+1]
            query_str_mode = 0
            print(
                    f"front 找到匹配：{number}，起始位置：{start_index}，结束位置：{end_index}")
            print(law_str)
        else:
            match_law_str_list = list(re.finditer(pattern_law_str, str_behand))
            match_law_str = match_law_str_list[-1]
            start_index = match_law_str.start()
            end_index = match_law_str.end()
            law_char = match_law_str.group()
            query_str_mode = 1
            if end_index < len(str_behand) + 1:
                law_str = str_behand[start_index - 1:start_index + 1]
            else:
                law_str = str_behand[start_index - 1:]
            print(
                f"behand 找到匹配：{number}，起始位置：{start_index}，结束位置：{end_index}")
            print(law_str)

        # sql query
        # cursor.execute("SELECT * FROM legal_provisions WHERE content LIKE ?",
        #                (f'第{number}%',))
        # rows = cursor.fetchall()

        cursor.execute(
            "SELECT * FROM legal_provisions WHERE type LIKE ? AND content LIKE ?",
            (f'%{law_str}%', f'第{number}%'))


        rows = cursor.fetchall()

        if len(rows) <= 0:
            print(f'type 不包涵{number}的数据')
            num_index_n = 3
            if query_str_mode == 0:
                if start_index <= num_index_n:
                    law_str = str_front[:start_index + 1]
                else:
                    law_str = str_front[start_index - num_index_n - 1:start_index + 1]
            else:
                if end_index < len(str_behand) + 1:
                    law_str = str_behand[start_index - num_index_n:start_index + 1]
                else:
                    law_str = str_behand[start_index - num_index_n:]
            cursor.execute(
                "SELECT * FROM legal_provisions WHERE title LIKE ? AND content LIKE ?",
                (f'%{law_str}%', f'第{number}%'))
            rows = cursor.fetchall()
        # 打印查询结果
        qa_prompt_tmpl_str = (
            "上下文信息如下。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "请根据上下文信息而不是先验知识来回答以下的查询。"
            "作为一个法律人工智能助手，你的回答要尽可能严谨。\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        context = ''
        print(len(rows))
        for idx, row in enumerate(rows):
            if idx > 5:
                break
            print(row)
            context += f'{row[4]}\n'

            # 打印每个字段的数据
            # print("Field 1:", field1)
            # print("Field 2:", field2)
            # print("Field 3:", field3)
            # print("Field 3:", field4)
            # print("Field 3:", field5)
        qa_prompt_tmpl = qa_prompt_tmpl_str.format(context_str=context, query_str=query)
        print(qa_prompt_tmpl)
        print('************')
        request_list = [{'query': qa_prompt_tmpl}]
        resp_list = inference_vllm(llm_engine, template, request_list)
        for request, resp in zip(request_list, resp_list):
            # print(f"query: {request['query']}")
            print('----------------')
            print(f"response: {resp['response']}")
    else:
        request_list = [{'query': query}]
        resp_list = inference_vllm(llm_engine, template, request_list)
        for request, resp in zip(request_list, resp_list):
            # print(f"query: {request['query']}")
            print('----------------')
            print(f"response: {resp['response']}")