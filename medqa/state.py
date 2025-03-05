from typing import TypedDict, List, Optional

class AnswerState(TypedDict):
    question: str
    round: int
    max_rounds: int
    answers: List[str] # Initial answers from each doctor agent
    feedback: List[str] # Feedback from the meta agent if inconsistency among doctors' answers is detected
    consensus_reached: bool
    final_answer: Optional[str]