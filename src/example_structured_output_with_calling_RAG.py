from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langgraph.graph import MessagesState, START, END
from crewai import LLM
from langchain_huggingface import ChatHuggingFace
from langgraph.prebuilt import create_react_agent
from typing import Literal, TypedDict, List, Dict, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command
from langgraph.graph import StateGraph
import json
import re
from typing_extensions import TypedDict
from dataclasses import dataclass
from typing import Optional


from pydantic import BaseModel, Field
from typing import List, Optional
from indexer import DocumentIndexer, HybridRetriever


############################################
# Define the LLM
############################################
from langchain_openai import AzureChatOpenAI
import httpx
import os

if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

HTTPX_CLIENT = httpx.Client(http2=True, verify='PATH_TO_CERTIFICATE')  # or False if you don't want to verify

llm = AzureChatOpenAI(
    azure_deployment="gpt-4-turbo",  # or your deployment
    api_version="2024-08-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    http_client=HTTPX_CLIENT,
    default_headers = {"X-Gravitee-Api-Key": os.getenv("OPENAI_API_KEY")},
    # other params...
)

PRECLINICAL_QUERY_DECOMPOSITION_PROMPT = """
"""


# Pydantic BaseModel to define the structure of the response
class PreclinicalQuery(BaseModel):
    """Preclinical query analysis and rewrite."""
    is_relevant: bool = Field(description="Whether the query is related to preclinical studies")
    reason: str = Field(description="Brief explanation why it is/isn't preclinical")
    sub_queries: Optional[List[str]] = Field( 
        description="List of rewritten sub-queries if the query is preclinical-related"
    )


# Pydantic BaseModel to define the structure of the response
class ClinicalQuery(BaseModel):
    """Clinical query analysis and rewrite."""
    is_relevant: bool = Field(description="Whether the query is related to preclinical studies")
    reason: str = Field(description="Brief explanation why it is/isn't preclinical")
    sub_queries: Optional[List[str]] = Field( 
        description="List of rewritten sub-queries if the query is preclinical-related"
    )


class WorkflowState(MessagesState):
    request: str                                        # One query as a whole
    pre_clinical_query: Optional[PreclinicalQuery]
    pre_clinical_chunks: Optional[List[Any]]
    retriever: Any                                      # We use HybridRetriever class, note here if we want our agent to do function call, we need to pass/save the function in the state


def preclinical_search_agent(state: WorkflowState):
    prompt = PRECLINICAL_QUERY_DECOMPOSITION_PROMPT.format(query=state['request'])
    # The result is a PreclinicalQuery object (pydantic BaseModel) containing sub-queries if the query is preclinical-related
    result = llm.with_structured_output(PreclinicalQuery).invoke(prompt)
    chunks = function(result, state['retriever'])

    return {"pre_clinical_query": result, "pre_clinical_chunks": None if len(chunks) == 0 else chunks} 


builder = StateGraph(WorkflowState)
builder.add_node(preclinical_search_agent)
builder.add_edge(START, "preclinical_search_agent")
builder.add_edge("preclinical_search_agent", END)
graph = builder.compile()

hybrid_retriever = HybridRetriever()

initial_state = WorkflowState(request="your query", retriever=hybrid_retriever)
graph.invoke(initial_state)