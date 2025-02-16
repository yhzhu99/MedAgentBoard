#########################################################################
#                                                                       #
#   Functions to initialize the local LLM model                         #
#                                                                       #
#########################################################################
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
        device_map="auto"
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

llm = initialize_local_model()

#########################################################################
#                                                                       #
#   Utility functions                                                   #
#                                                                       #
#########################################################################

def extract_json_deepseek_r1(text):
    # Remove think tags and content
    import re
    text_without_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Extract JSON content
    json_match = re.search(r'```json\s*(.*?)\s*```', text_without_think, flags=re.DOTALL)
    
    if json_match:
        return json_match.group(1)
    return None


#########################################################################
#                                                                       #
#   Agent prompt design                                                 #
#                                                                       #
#########################################################################

DPO_PROMPT = """

"""

TOXICITY_PROMPT = """

"""

CATEGORIZATION_PROMPT = """

"""

DPO_AGENT_ROLE = """

"""

TOXICOLOGY_AGENT_ROLE = """
You are a specialized toxicologist by training.
"""

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

from langgraph.graph import MessagesState, START, END
# from crewai import LLM
# from langchain_huggingface import ChatHuggingFace
from langgraph.prebuilt import create_react_agent
from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from typing import Literal, TypedDict, List, Union
from langgraph.graph import MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command
from langgraph.graph import StateGraph
import json
from typing_extensions import TypedDict


#########################################################################
#                                                                       #
#   Agent design                                                        #
#                                                                       #
#########################################################################

# Assuming llm initialization remains the same
llm = initialize_local_model()

# Only need for iteration/multiple rounds collobration
def get_next_node(current_node: str, findings: dict) -> str:
    if current_node == "dpo_agent":
        return "toxicology_agent"
    elif current_node == "toxicology_agent":
        return "categorization_agent"
    elif current_node == "categorization_agent":
        return END
    return END



# Regulatory Agent
dpo_agent = create_react_agent(
    llm,
    tools=[],
    state_modifier=make_system_prompt(DPO_AGENT_ROLE)
)

# Toxicology Agent
toxicology_agent = create_react_agent(
    llm,
    tools=[],
    state_modifier=make_system_prompt(TOXICOLOGY_AGENT_ROLE),
    # state_modifier: system prompt | the overall prompt structure is [state_modifier, prompt in agent.invoke()]
)

# Categorization Agent
categorization_agent = create_react_agent(
    llm,
    tools=[],
)

from typing import Literal, TypedDict, List, Dict, Any

class WorkflowState(MessagesState):
    document: str
    dpo_present: bool
    dpo_findings: Dict[str, Any]
    toxicity_findings: Dict[str, Any]
    final_category: Dict[str, Any]

def dpo_node(state: WorkflowState) -> dict:
    # Format the DPO prompt with the document
    formatted_prompt = DPO_PROMPT.format(report=state["document"])
    
    # Create a new message with the formatted prompt
    prompt_message = {"role": "user", "content": formatted_prompt}
    
    result = dpo_agent.invoke({"messages": [prompt_message]})
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="dpo_agent"
    )
    try:
        content = result["messages"][-1].content
        findings = json.loads(extract_json_deepseek_r1(content))
        dpo_present = findings.get('dpo_present', 'No').lower() == 'yes'
        del findings['dpo_present']
    except (json.JSONDecodeError, KeyError, AttributeError):
        findings = {"error": "Invalid JSON response"}
        dpo_present = False
    return Command(
        update={
            "messages": result["messages"],
            "dpo_findings": findings,
            "dpo_present": dpo_present
        },
        goto="toxicology_agent",
    )

def toxicology_node(state: WorkflowState) -> dict:
    # Format the DPO prompt with the document
    formatted_prompt = TOXICITY_PROMPT.format(report=state["document"])
    # Create a new message with the formatted prompt
    prompt_message = {"role": "user", "content": formatted_prompt}
    
    result = toxicology_agent.invoke({"messages": [prompt_message]})
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="toxicology_agent"
    )
    try:
        content = result["messages"][-1].content
        findings = json.loads(extract_json_deepseek_r1(content))
    except (json.JSONDecodeError, KeyError):
        findings = {"error": "Invalid JSON response"}

    return Command(
        update={
            "messages": result["messages"],
            "toxicity_findings": findings
        },
        goto="categorization_agent",
    )

def categorization_node(state: WorkflowState) -> dict:
    if state["dpo_present"]:
        formatted_prompt = CATEGORIZATION_PROMPT.format(
            dpo_findings=json.dumps(state["dpo_findings"]),
            toxicity_analysis=json.dumps(state["toxicity_findings"])
        )
    else:
        formatted_prompt = CATEGORIZATION_PROMPT.format(
            dpo_findings="No findings",
            toxicity_analysis=json.dumps(state["toxicity_findings"])
        )
    prompt_message = {"role": "user", "content": formatted_prompt}
    result = categorization_agent.invoke({
        "messages": [prompt_message],
    })
    print(result["messages"])
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="category_agent"
    )
    try:
        final_category = json.loads(extract_json_deepseek_r1(result["messages"][-1].content))
    except json.JSONDecodeError:
        final_category = {"error": "Invalid JSON response"}

    return Command(
        update={
            "messages": result["messages"],
            "final_category": final_category
        },
        goto=END,
    )

# Graph Configuration
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("dpo_agent", dpo_node)
workflow.add_node("toxicology_agent", toxicology_node)
workflow.add_node("categorization_agent", categorization_node)

# Add edges
workflow.add_edge(START, "dpo_agent")
workflow.add_edge("dpo_agent", "toxicology_agent")
workflow.add_edge("toxicology_agent", "categorization_agent")
workflow.add_edge("categorization_agent", END)

# Compile the graph
graph = workflow.compile()

def analyze_pharmaceutical_document(document: str) -> WorkflowState:
    initial_state = {
        "document": document,  # Store the document in the state
        "messages": [],  # No need to store messages in state
        "dpo_findings": {},
        "toxicity_findings": {},
        "dpo_present": False,
        "final_category": {}
    }
    return graph.invoke(initial_state)


#########################################################################
#                                                                       #
#   Usage                                                               #
#                                                                       #
#########################################################################

# Visualizations
# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

# sample_doc = """
# Clinical trial report for Drug X:
# - Severe hepatotoxicity observed in 30% of patients
# - FDA issued a DPO compliance notice due to data integrity issues
# - Trial discontinued following DSMB recommendation
# """

# # Run the analysis pipeline
# results = analyze_pharmaceutical_document(sample_doc)

# # Interpret results
# print("Final Category:")
# print(json.dumps(results["final_category"], indent=2))

# print("\nDPO Findings:")
# print(json.dumps(results["dpo_findings"], indent=2))

# print("\nToxicity Findings:")
# print(json.dumps(results["toxicity_findings"], indent=2))