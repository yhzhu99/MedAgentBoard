from langgraph.graph import MessagesState
from typing import Literal, TypedDict, List, Dict, Any, Optional

# 暂定的全局变量
class WorkflowState(MessagesState):
    request: str          # 用户原始请求
    code: Optional[str]   # 生成的代码
    code_output: Optional[str]  # 代码运行结果
    error: Optional[str]  # 执行错误信息
    attempts: int         # 已尝试次数
    data: Any        # 数据（类型暂定为<class 'pandas.core.frame.DataFrame'>）