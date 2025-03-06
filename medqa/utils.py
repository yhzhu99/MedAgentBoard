from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from crewai import LLM
import os
import logging


from medqa.state import AnswerState

load_dotenv() # Load environment variables from .env file

def initialize_ds_model(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        base_url="https://api.deepseek.com/v1", 
        api_key=os.getenv("DEEPSEEK_API_KEY")   # Get your API key from environment variables
    )
    
def initialize_gpt_model(model_name: str) -> ChatOpenAI:
    os.environ["OPENAI_API_BASE"] = "https://api.gptsapi.net/v1"
    os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")
    return ChatOpenAI(model=model_name)

def initialize_qwen_model(model_name: str) -> ChatOpenAI:
    return LLM(
        model=model_name,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", 
        api_key=os.getenv("QWEN_API_KEY") 
    )
    
def log_state_change(old_state: AnswerState, new_state: AnswerState):
    """
    Logs the changes between the old state and the new state.
    """
    for key in new_state:
        if old_state.get(key) != new_state[key]:
            logging.info(f"State change: {key} -> {new_state[key]}")