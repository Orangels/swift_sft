import logging
import sys
import os
import torch
from llama_index.core import PromptTemplate, Settings, StorageContext, \
    load_index_from_storage
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, \
    VectorStoreIndex, load_index_from_storage, \
    StorageContext, QueryBundle
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# 定义日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 定义system prompt
SYSTEM_PROMPT = """你是一个法律人工智能助手。"""
query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

# 定义qa prompt
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
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# 定义refine prompt
refine_prompt_tmpl_str = (
    "原始查询如下：{query_str}"
    "我们提供了现有答案：{existing_answer}"
    "我们有机会通过下面的更多上下文来完善现有答案（仅在需要时）。"
    "------------"
    "{context_msg}"
    "------------"
    "考虑到新的上下文，优化原始答案以更好地回答查询。 如果上下文没有用，请返回原始答案。"
    "Refined Answer:"
)
refine_prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)

# 使用llama-index创建本地大模型
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name='/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v8-20240424-190644/checkpoint-4000-merged',
    model_name='/home/zwmx/xm_dev/ls_project/llm_sft_output/qwen1half-14b-chat/v8-20240424-190644/checkpoint-4000-merged',
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16},
)
Settings.llm = llm

# 使用LlamaDebugHandler构建事件回溯器，以追踪LlamaIndex执行过程中发生的事件
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

# 使用llama-index-embeddings-huggingface构建本地embedding模型
Settings.embed_model = HuggingFaceEmbedding(
    # model_name="/home/zwmx/xm_dev/ls_dev/model/bge-base-zh-v1.5"
    model_name="/home/zwmx/xm_dev/ls_project/models/bge-base-zh-v1.5"
)

# 从存储文件中读取embedding向量和向量索引
storage_context = StorageContext.from_defaults(persist_dir="doc_emb")
index = load_index_from_storage(storage_context)

# 构建查询引擎
query_engine = index.as_query_engine(similarity_top_k=5)

# 输出查询引擎中所有的prompt类型
prompts_dict = query_engine.get_prompts()
print(list(prompts_dict.keys()))

# 更新查询引擎中的prompt template
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl,
     "response_synthesizer:refine_template": refine_prompt_tmpl}
)
n = -1
while True:
    query_str = input('请输入: ')

    # 查询获得答案
    # response = query_engine.query(
    #     "网购商品用快递送达，商品在快递途中、签收之前毁损的风险谁承担？")

    response = query_engine.query(query_str)

    n += 1

    print('!!!!!!!!')
    print(response)
    print('!!!!!!!!')

    # 输出formatted_prompt
    event_pairs = llama_debug.get_llm_inputs_outputs()
    print(event_pairs[n][1].payload["formatted_prompt"])
