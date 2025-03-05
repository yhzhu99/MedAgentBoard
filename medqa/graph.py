from langgraph.graph import StateGraph
from medqa.state import AnswerState
from medqa.nodes import (
    generate_answers_node,
    check_consensus_node,
    generate_feedback_node,
    refine_answers_node,
    final_node
)

class MedQAGraph:
    def __init__(self, question: str, max_rounds: int = 5):
        self.graph = StateGraph(AnswerState)
        self.question = question
        self.max_rounds = max_rounds
        
    def build_graph(self):
        self.graph.add_node("generate_answers", generate_answers_node)
        self.graph.add_node("check_consensus", check_consensus_node)
        self.graph.add_node("generate_feedback", generate_feedback_node)
        self.graph.add_node("refine_answers", refine_answers_node)
        self.graph.add_node("final", final_node)
        
        self.graph.set_entry_point("generate_answers")
        
        # Define edges
        self.graph.add_edge("generate_answers", "check_consensus")
        self.graph.add_edge("generate_feedback", "refine_answers")
        self.graph.add_edge("refine_answers", "check_consensus")
        
        # Conditional edges from check_consensus
        self.graph.add_conditional_edges(
            "check_consensus",
            # Go to "final" node if consensus is reached or max rounds are reached, otherwise go to "generate_feedback" node
            lambda state: "final" if state['consensus_reached'] or state['round'] >= state['max_rounds'] else "generate_feedback"
        )
        
        return self.graph.compile()
    
    def run(self):
        initial_state = {
            'question': self.question,
            'round': 0,
            'max_rounds': self.max_rounds,
            'answers': [],
            'feedback': [],
            'consensus_reached': False,
            'final_answer': None
        }
        
        app = self.build_graph()
        result = app.invoke(initial_state)
        return result['final_answer']