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

# TODO: Modified the file name of logs for different questions in later experiments
def log_state_change(old_state, new_state):
    """Log changes to state to track the flow of multi-agent collaboration."""
    
    # Initialize log file if this is the first round
    if old_state.get('round', 0) == 0:
        with open('collaboration_log.txt', 'w') as f:
            f.write(f"<Question>: {new_state['question']}\n\n")
    
    # Log new answers
    if old_state['round'] != new_state['round']:
        with open('collaboration_log.txt', 'a') as f:
            f.write(f"<Round {new_state['round']}>:\n")
            f.write("\nAnswers from each agent:\n\n")
            for i, answer in enumerate(new_state['current_answers']):
                f.write(f"Doctor {i+1}: {answer}\n\n")
    
    # Log consensus status
    if old_state['consensus_reached'] != new_state['consensus_reached']:
        with open('collaboration_log.txt', 'a') as f:
            f.write(f"\n<Consensus status after Round {new_state['round']}>: {new_state['consensus_reached']}\n\n")
    
    # Log feedback
    if old_state['feedback'] != new_state['feedback']:
        with open('collaboration_log.txt', 'a') as f:
            f.write(f"\n<Feedback summary from Round {new_state['round']}>:\n")
            f.write(f"{new_state['feedback'][-1]}\n\n")
    
    # Log final answer
    if old_state['final_answer'] != new_state['final_answer']:
        with open('collaboration_log.txt', 'a') as f:
            f.write("\n<Final answer>:\n")
            f.write(f"{new_state['final_answer']}\n\n")