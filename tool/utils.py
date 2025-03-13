from langchain.chat_models import ChatOpenAI
import os
import re
from .state import WorkflowState

def initialize_gpt_model(model_name):
    # 自用小模型，仅用于小样本测试
    os.environ["OPENAI_API_BASE"] = "https://api.gptsapi.net/v1"
    os.environ["OPENAI_API_KEY"] = "sk-2Cd64920f5df1880606c597a75f9f05009d2d9222a4Jur6C"
    from langchain.chat_models import ChatOpenAI
    return ChatOpenAI(model=model_name)

def extract_code_from_response(content):
    content = str(content)  # 将<class 'crewai.crews.crew_output.CrewOutput'>变成str
    code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
    clean_code = code_match.group(1).strip() if code_match else content.strip()
    return clean_code

def initialize_workflow_state(user_request, data):
    return WorkflowState(
        request=user_request,
        code=None,
        code_output=None,
        error=None,
        attempts=0,
        data = data,
    )

