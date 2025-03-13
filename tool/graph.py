from langgraph.graph import StateGraph, START, END
from .state import WorkflowState
from .nodes import CodeNodes

def build_graph():
    nodes = CodeNodes()
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node('code_generator', nodes.code_generator)
    workflow.add_node('code_executor', nodes.code_executor)
    workflow.add_node('code_auto_reviewer', nodes.code_auto_reviewer)
    workflow.add_node('code_reviewer', nodes.code_reviewer)
    
    
    workflow.add_edge(START, 'code_generator')
    workflow.add_edge('code_generator', 'code_executor')
    workflow.add_edge('code_executor', 'code_auto_reviewer')
    workflow.add_edge('code_auto_reviewer', 'code_reviewer')
    workflow.add_edge('code_reviewer', END)
    # workflow.add_conditional_edges(
    #     "code_reviewer",
    #     nodes.review_decision,
    #     {
    #         "retry": "code_generator",
    #         "end": END
    #     }
    # )
    
    return workflow.compile()