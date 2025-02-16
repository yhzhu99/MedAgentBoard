from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def initialize_local_model(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    max_length: int = 4096,
    temperature: float = 0.1,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15
) -> HuggingFacePipeline:
    """
    Initialize a local Hugging Face model for use with LangChain
    """
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        # use_flash_attention_2 = False, # 尝试禁用flash attention以兼容windows
    )

    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        truncation=True,
    )

    # Create LangChain wrapper
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def initialize_gpt_model(model_name):
    import os
    os.environ["OPENAI_API_BASE"] = "https://api.gptsapi.net/v1"
    os.environ["OPENAI_API_KEY"] = "sk-2Cd64920f5df1880606c597a75f9f05009d2d9222a4Jur6C"
    from langchain.chat_models import ChatOpenAI
    return ChatOpenAI(model=model_name)

from graph import WorkflowState

def initialize_workflow_state(user_request, data) -> WorkflowState:
    return WorkflowState(
        request=user_request,
        code=None,
        code_output=None,
        error=None,
        attempts=0,
        data = data,
    )

import re

def extract_code_from_response(content):
    code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
    clean_code = code_match.group(1).strip() if code_match else content.strip()
    return clean_code

def make_system_prompt(suffix: str) -> str:
    # System prompt for the agent, in another saying, it is the prompt to describe the agent
    # For dpo_agent, 
    # "You are a specialized assistant, only focusing on analyzing Data Protection Ordinance decisions from provided documents.
    # You answer will be used to identify key elements and potential reason of drug development discontinuation reason from DPO decisions. You
    # are working with other toxicologists to provide a comprehensive supportive information to an expert who will categorize the discountinuation reason."
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        "will help where you left off. Execute what you can to make progress."
        f"\n{suffix}"
    )

import pickle

def load_data(pth = '/home/kisara/RESEARCH/1/src/datasets/mimic-datasets/mimic3_all/ts_note_all.pkl'):
    with open(pth, 'rb') as f:
        merged_data = pickle.load(f)
    return merged_data

import csv
import pandas as pd

def load_raw_data(pth = '/home/kisara/RESEARCH/1/src/datasets/mimic-datasets/mytest.csv'):
    df = pd.read_csv(pth, dtype=str)
    return df
    