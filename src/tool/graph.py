from langgraph.prebuilt import create_react_agent
from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.types import Command
from langgraph.graph import MessagesState, START, END

from langgraph.graph import StateGraph
import json
from typing import Literal, TypedDict, List, Dict, Any, Optional

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langgraph.checkpoint.memory import MemorySaver

# from utils import *
from .utils import *
from .prompt import *
from .state import WorkflowState

# # 暂定的全局变量
# class WorkflowState(MessagesState):
#     request: str          # 用户原始请求
#     code: Optional[str]   # 生成的代码
#     code_output: Optional[str]  # 代码运行结果
#     error: Optional[str]  # 执行错误信息
#     attempts: int         # 已尝试次数
#     data: Any        # 数据（类型暂定为<class 'pandas.core.frame.DataFrame'>）

# TODO 下面这段代码直接放到了全局环境中，如果该文件被重复引用可能会导致不必要的初始化等问题？
# 初始化大语言模型
llm = initialize_gpt_model('gpt-4o')

# 定义各项agent
code_generator_agent = create_react_agent(
    llm,
    tools=[],
    # state_modifier =  # agent role prompt
    checkpointer=MemorySaver()  # 添加记忆
)
code_reviewer_agent = create_react_agent(
    llm,
    tools=[],
    # state_modifier =  # agent role prompt
)

def code_generator(state: WorkflowState) -> dict:
    # 根据不同attempt次数设计不同的prompt
    if state['attempts'] == 0:
        if state['data'] is not None:
            # prompt_input = code_generator_prompt2.format(request = state['request'], data_type = str(type(state['data'])), data_example = state['data'].to_string(max_rows=None, max_cols=None, max_colwidth=None)[:5000])
            prompt_input = code_generator_prompt2.format(request = state['request'], data_type = str(type(state['data'])), data_example = str(state['data']))
        else:
            prompt_input = code_generator_prompt.format(request = state['request'])
        print('prompt_input:', prompt_input)
        
    elif state['attempts'] >=1:
        prompt_input = code_generator_retry_prompt.format(error = state['error'], request = state['request'])
    
    config = {"configurable": {"thread_id": "thread-1"}}    # 标记线程
    inputs = {"messages": [("user", prompt_input)]}
    response = None    # 储存message 
    for s in code_generator_agent.stream(inputs, config, stream_mode="values"):     # 这个for循环是什么意思：第一个是human_request content，第二个是llm_response content
        message = s["messages"][-1]
        # if isinstance(message, tuple):
        #     print(message)
        # else:
        #     message.pretty_print()
        message.pretty_print()
        response = message
    
    # 从响应中提取代码
    clean_code = extract_code_from_response(response.content)
    # print(clean_code)
    
    # 更新尝试次数
    new_attempts = state["attempts"] + 1
    
    return Command(
        update={
            "code": clean_code,
            "attempts": new_attempts
        },
        goto="code_executor",
    )


def code_executor(state: WorkflowState) -> dict:
    code = state["code"]
    error = None
    output = None
    try:
        data = state['data']
        
        result = {}
        # 在隔离环境中执行代码
        exec(code, {'data': data}, result)   # 第二个参数为代码运行环境，用于向内传递参数；第三个参数用于收集代码内变量
        output = result['result']
        print('code output: ', output)
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"
        print('error: ',error)
    
    return Command(
        update = {"error": error, 'code_output': output},
        goto = "code_reviewer"
    )


def code_reviewer(state: WorkflowState):
    # 最大尝试次数设为3次
    MAX_ATTEMPTS = 3
    
    # 检查是否存在报错
    if state["error"] and state["attempts"] < MAX_ATTEMPTS:
        print(f"检测到错误，尝试重新生成（剩余尝试次数：{MAX_ATTEMPTS - state['attempts']}）")
        return Command(
            update = {},
            goto = "code_generator"
        )
    
    # 检查答案形式是否合理  TODO
    # prompt_input = code_reviewer_prompt.format(request = state['request'], output = state['code_output'])
    # result = code_reviewer_agent.invoke({"messages": [("user", prompt_input)]})
    # response = result["messages"][-1].content
    # print('ttttt:',response)
    # if response=='N':
        
    
    return Command(
        update = {},
        goto = END
    )   
    
    
def meet_demand_reviewer(state: WorkflowState):
    pass
     

def build_graph():
    
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node('code_generator', code_generator)
    workflow.add_node('code_executor', code_executor)
    workflow.add_node('code_reviewer', code_reviewer)
    
    workflow.add_edge(START, 'code_generator')
    workflow.add_edge('code_generator', 'code_executor')
    workflow.add_edge('code_executor', 'code_reviewer')
    workflow.add_edge('code_reviewer', END)
    
    graph = workflow.compile()
    
    return graph

def run_graph(user_request, data=None):
    graph = build_graph()
    initial_state = initialize_workflow_state(user_request = user_request, data = data)
    result = graph.invoke(initial_state)    # 真正调用时再决定输入的状态
    print(result)
    return result   # xxx
    

if __name__ == "__main__":
    
    graph = build_graph()
    
    test_rq = "What is the highest temperature of the patient?"
    test_data = load_raw_data()
    initial_state = initialize_workflow_state(user_request = test_rq, data = test_data)
    
    result = graph.invoke(initial_state)    # 真正调用时再决定输入的状态
    
    
    
    