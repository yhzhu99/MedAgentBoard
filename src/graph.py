from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from tool.graph import run_graph

from langgraph.prebuilt import create_react_agent
from prompt import *

class GlobalState(TypedDict):
    user_request: str
    need_code: Optional[str]
    
    code_request: Optional[str]
    code_data: Optional[str]
    
    code_execute_output: Optional[str]
    code_execute_error: Optional[str]

def call_code_graph(state: GlobalState):  
    
    code_execute_state = run_graph(state['code_request'], state['code_data'])
    
    print('\nCode execution done.\n')
    
    return Command(
        update = {
            "code_execute_output": code_execute_state['code_output'],
            "code_execute_error": code_execute_state['error']
        },
        # goto = END
    )

# TODO 需要打包
import os
os.environ["OPENAI_API_BASE"] = "https://api.gptsapi.net/v1"
os.environ["OPENAI_API_KEY"] = "sk-2Cd64920f5df1880606c597a75f9f05009d2d9222a4Jur6C"
from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model='gpt-3.5-turbo')
llm = ChatOpenAI(model='gpt-4o')
judge_agent = create_react_agent(
    llm,
    tools=[],
    response_format=judge_output_format, # 在prebuilt react agent中限制输出格式：https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-structured-output/
)

# NotImplementedError: https://blog.csdn.net/lovechris00/article/details/142987440

# from API_CONFIG import load_llm
# # llm = load_llm(model_name='deepseek-v3-official')
# llm = load_llm(model_name='qwen-max-latest' )
# judge_agent = create_react_agent(
#     llm,    # 限制输出格式
#     tools=[],
#     response_format=judge_output_format
# )

def judge_code_need(state: GlobalState):
    # TODO 怎么让大模型知道数据情况？（用户是如何输入数据的？（提交任意文件？llm是否需要反馈））
    # 将用户需求转化为更易于llm理解的形式(user_request->code_request)
    
    # prompt_input = "请严格按照下面的要求操作：\n1.阅读用户需求，判断是否需要生成一段python代码用于辅助解决问题；\n2.返回一个单词：'Yes'或'No'，'Yes'代表需要生成代码，'No'代表不需要生成代码。\n3.如果需要生成代码，将用户的代码需求转化为更精炼的形式。\n\n用户需求：{}\n\n其中，用户提供了一份EHR数据，数据的前5000个字符如下：{}".format(state['user_request'], state['code_data'].to_string(max_rows=None, max_cols=None, max_colwidth=None)[:5000])
    prompt_input = judge_prompt_2.format(request = state['user_request'], data = state['code_data'].to_string(max_rows=None, max_cols=None, max_colwidth=None)[:5000])
    
    result = judge_agent.invoke({"messages": [("user", prompt_input)]})
    
    print('\n\n\n\n', result,'\n\n\n\n')
    
    # response = result["messages"][-1].content
    response = result['structured_response']    # 支持的模型列表：https://www.langchain.com.cn/docs/integrations/chat/  # format直接提示：https://www.langchain.com.cn/docs/how_to/structured_output/
    print('\n\n\n\nresponse:', response)    # TODO 如果response不是yes或no的处理
    
    return Command(
        update = {
            "need_code": response.need_code,
            'code_request': response.code_request
        },
    )

"""
    gpt3.5一个错误案例
    AIMessage(content='Yes\nUser requirement: "What is the highest temperature of the patient?"\n\nRefined code requirement: Find the highest temperature value in the EHR data.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 31, 'prompt_tokens': 1042, 'total_tokens': 1073, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_0165350fbb', 'finish_reason': 'stop', 'logprobs': None}, id='run-76a550a1-9f2d-4cb9-9ffe-32b784fa11c4-0', usage_metadata={'input_tokens': 1042, 'output_tokens': 31, 'total_tokens': 1073, 'input_token_details': {}, 'output_token_details': {}})], 'structured_response': judge_output_format(need_code='No', code_request=None)}
"""
"""
    gpt3.5一个错误案例
    AIMessage(content='Yes\n用户需要提取患者的体温数据以获取最高体温。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 1042, 'total_tokens': 1065, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_0165350fbb', 'finish_reason': 'stop', 'logprobs': None}, id='run-77c5c680-ead0-4561-9651-0e42b789f8b2-0', usage_metadata={'input_tokens': 1042, 'output_tokens': 23, 'total_tokens': 1065, 'input_token_details': {}, 'output_token_details': {}})], 'structured_response': judge_output_format(need_code='No', code_request=None)}
"""

def build_graph():
    
    workflow = StateGraph(GlobalState)
    
    # TODO 无条件边
    workflow.add_node('Code', call_code_graph)
    workflow.add_node('Judge', judge_code_need)
    
    workflow.add_edge(START, 'Judge')
    # workflow.add_edge('Judge', 'Code')
    workflow.add_conditional_edges(
        'Judge',
        lambda state: state['need_code'].lower(),
        {
            'yes': 'Code',
            'no': END
        }
    )
    
    workflow.add_edge('Code', END)
    
    graph = workflow.compile()
    
    return graph


def load_raw_data(pth = '/home/kisara/RESEARCH/1/src/datasets/mimic-datasets/mytest.csv'):
    import pandas as pd
    df = pd.read_csv(pth, dtype=str)
    return df

if __name__=='__main__':
    
    graph = build_graph()
    initial_state = GlobalState(
        user_request = "What is the highest temperature of the patient?",
        # user_request = "Which is the best hospital in America?",
        # code_request = "What is the highest temperature of the patient?",
        
        code_data = load_raw_data() # 暂定为在这里导入数据
    )
    
    result = graph.invoke(initial_state)
    
    print('Need code?:', result['need_code'])
    print('Code execution output:', result['code_execute_output'])
    print('Code error:', result['code_execute_error'])
    
    
    