from typing import TypedDict, List, Optional, Dict

class AnswerState(TypedDict):
    question: str
    round: int
    max_rounds: int
    previous_answers: List[str] # previous-round answers from each doctor agent
    current_answers: List[str] # current-round answers from each doctor agent
    feedback: str # Feedback from the meta agent if inconsistency among doctors' answers is detected
    consensus_reached: bool
    final_answer: Optional[str]
    log: Dict[any, any] # Log of collaboration