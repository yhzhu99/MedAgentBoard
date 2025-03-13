from .graph import build_graph
from .state import WorkflowState
from .utils import initialize_workflow_state

def inference(user_request: str, data):
    
    # 构建工作流
    workflow = build_graph()
    
    # 初始化状态
    initial_state = initialize_workflow_state(
        user_request=user_request,
        data=data
    )
    
    # 执行工作流
    result = workflow.invoke(initial_state)
    print("\n" + "="*40)
    print("原始请求:", user_request)
    print("生成代码:\n", result['code'])
    print("执行结果:", result['code_output'])
    print("错误信息:", result['error'])
    print("尝试次数:", result['attempts'])
    print("="*40 + "\n")
    return result


if __name__ == "__main__":
    inference('找出列表中所有合法数字中的最大值', data=[5,32,12,41,111,0,90,'NA'])
    
    
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Literal, TypedDict, List, Dict, Any, Optional

import json

class CodeToolInput2(BaseModel):
    """Input schema for running workflow graph."""
    user_request: str = Field(..., description="The user's request or problem to solve")
    data_path: str = Field(..., description="Data path needed to process the request")

class CodeTool2(BaseTool):
    name: str = "Code Tool"
    description: str = "A tool that generates code to solve a given problem"
    args_schema: Type[BaseModel] = CodeToolInput2
    def _run(self, user_request: str, data_path: str) -> str:
        """Execute the workflow graph with provided inputs"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # 这里将数据类型固定为json了，可能需要根据数据集不同而修改
        
        result = inference(user_request=user_request, data=data)
        
        # 处理结果：包括result['code_output']和result['error']
        if result['error']:
            return "Error when executing code."
        else:
            return str(result['code_output'])  
