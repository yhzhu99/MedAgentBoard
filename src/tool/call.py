from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Literal, TypedDict, List, Dict, Any, Optional

from tool.graph import run_graph

class CodeToolInput(BaseModel):
    """Input schema for running workflow graph."""
    user_request: str = Field(..., description="The user's request or problem to solve")
    data: Any = Field(..., description="Contextual data needed to process the request")

class CodeTool(BaseTool):
    name: str = "Code Tool"
    description: str = "A tool that generates code to solve a given problem"
    args_schema: Type[BaseModel] = CodeToolInput

    def _run(self, user_request: str, data: Any) -> str:
        """Execute the workflow graph with provided inputs"""

        result = run_graph(user_request=user_request, data=data)
        
        # 处理结果：包括result['code_output']和result['error']
        if result['error']:
            return "Error when executing code"
        else:
            return str(result['code_output'])   

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
        
        result = run_graph(user_request=user_request, data=data)
        
        # 处理结果：包括result['code_output']和result['error']
        if result['error']:
            return "Error when executing code."
        else:
            return str(result['code_output'])   
            