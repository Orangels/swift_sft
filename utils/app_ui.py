from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Iterator
import json
import os
import re
import sqlite3
from cn2an import cn2an, an2cn
import traceback

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)

import gradio as gr

History = List[Union[Tuple[str, str], List[str]]]


def gradio_chat_demo() -> None:
    ckpt_dir = '/home/zwmx/xm_dev/ls_project/swift/scripts/ls/output/qwen1half-14b-chat/v8-20240427-133147/checkpoint-93-merged'
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

    def clear_session() -> History:
        return []


    def model_chat(query: str, total_history: str) -> Iterator[Tuple[str, History]]:
        try:
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
            # 0 front, 1 behand
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
                match_law_str_list = list(
                    re.finditer(pattern_law_str, str_front))

                if match_law_str_list:
                    match_law_str = match_law_str_list[-1]
                    start_index = match_law_str.start()
                    end_index = match_law_str.end()
                    law_char = match_law_str.group()
                    law_str = str_front[start_index - 1:start_index + 1]
                    query_str_mode = 0
                    print(
                        f"front 找到匹配：{number}，起始位置：{start_index}，结束位置：{end_index}")
                    print(law_str)
                else:
                    match_law_str_list = list(
                        re.finditer(pattern_law_str, str_behand))
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
                conn = sqlite3.connect('../sqlite/law.db')
                cursor = conn.cursor()

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
                            law_str = str_front[
                                      start_index - num_index_n - 1:start_index + 1]
                    else:
                        if end_index < len(str_behand) + 1:
                            law_str = str_behand[
                                      start_index - num_index_n:start_index + 1]
                        else:
                            law_str = str_behand[start_index - num_index_n:]
                    cursor.execute(
                        "SELECT * FROM legal_provisions WHERE title LIKE ? AND content LIKE ?",
                        (f'%{law_str}%', f'第{number}%'))
                    rows = cursor.fetchall()

                conn.close()

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
                qa_prompt_tmpl = qa_prompt_tmpl_str.format(context_str=context,
                                                           query_str=query)
                print(qa_prompt_tmpl)
                print('************')
                request_list = [{'query': qa_prompt_tmpl}]
                resp_list = inference_vllm(llm_engine, template, request_list)
                history = [(query, resp_list[0]['response'])]
                total_history = total_history + history
                yield '', total_history
                # return '', resp_list[0]['history']
                # for request, resp in zip(request_list, resp_list):
                #     # print(f"query: {request['query']}")
                #     print('----------------')
                #     print(f"response: {resp['response']}")
                #     return resp['response']
            else:
                request_list = [{'query': query}]
                resp_list = inference_vllm(llm_engine, template, request_list)
                history = resp_list[0]['history']
                total_history = total_history + history
                yield '', total_history
                # return '', resp_list[0]['history']
                # for request, resp in zip(request_list, resp_list):
                #     # print(f"query: {request['query']}")
                #     print('----------------')
                #     print(f"response: {resp['response']}")
                #     return resp['response']
        except Exception as e:
            print('***********')
            print(e)
            traceback.print_exc()
            print('***********')

            request_list = [{'query': query}]
            resp_list = inference_vllm(llm_engine, template, request_list)
            history = resp_list[0]['history']
            total_history = total_history + history
            yield '', total_history


    with gr.Blocks() as demo:
        # gr.Markdown(f'<center><font size=8>{model_name} Bot</center>')
        gr.Markdown(f'<center><font size=8>MetaAvatar Bot</center>')

        # chatbot = gr.Chatbot(label=f'{model_name}')
        chatbot = gr.Chatbot(label='MetaAvatar-Chat-14B')
        message = gr.Textbox(lines=2, label='Input')

        with gr.Row():
            clear_history = gr.Button('🧹 清除历史对话')
            send = gr.Button('🚀 发送')
        send.click(
            model_chat, inputs=[message, chatbot], outputs=[message, chatbot])
        clear_history.click(
            fn=clear_session, inputs=[], outputs=[chatbot], queue=False)
    # Compatible with InferArguments

    demo.queue().launch(
        height=1000,
        share=False,
        server_name='0.0.0.0',
        server_port=7860)


if __name__ == "__main__":
    gradio_chat_demo()
