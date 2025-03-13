from langgraph.graph import MessagesState
from typing import Optional, Any

class WorkflowState(MessagesState):
    request: str
    code: Optional[str]
    code_output: Optional[str]
    error: Optional[str]
    attempts: int
    data: Any