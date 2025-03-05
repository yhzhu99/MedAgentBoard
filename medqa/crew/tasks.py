from crewai import Task
from typing import List
from medqa.crew.agents import MedQAAgents

class MedicalTasks:
    def __init__(self, agents: MedQAAgents):
        self.agents = agents
        
    # Define the initial answer task handled each doctor agent
    def initial_answer_task(self) -> List[Task]:
        return [
            Task(
                description="{question}",
                agent=self.agents.cardiologist(),
                expected_output="One single answer and choice with brief reasoning and explanation for the choice"
            ),
            Task(
                description="{question}",
                agent=self.agents.neurologist(),
                expected_output="One single answer and choice with brief reasoning and explanation for the choice"
            ),
            Task(
                description="{question}",
                agent=self.agents.general_physician(),
                expected_output="One single answer and choice with brief reasoning and explanation for the choice"
            )
        ]
    
    # Define the feedback task handled by the meta agent
    def feedback_task(self) -> Task:
        return Task(
            description="Analyze these answers and identify key disagreements:\n\n{'='*20}\n" +
                        "\n\n".join([f"Doctor {i+1}: {ans}" for i, ans in enumerate(state['answers'])]),
            agent=self.agents.meta_agent(),
            expected_output="Summary of conflicting opinions and suggested focus areas for next round of collaboration and refinement to reach a consensus among the doctors"
        )
    
    # Define the refinement task handled by each doctor agent given the feedback
    def refinement_task(self, feedback: str) -> List[Task]:
        return [
            Task(
                description=f"{{question}}\n\nPrevious feedback:\n{feedback}", # TODO: Add previous answers?
                agent=self.agents.cardiologist(),
                expected_output="Revised answer (single choice) according to the feedback"
            ),
            Task(
                description=f"{{question}}\n\nPrevious feedback:\n{feedback}",
                agent=self.agents.neurologist(),
                expected_output="Revised answer (single choice) according to the feedback"
            ),
            Task(
                description=f"{{question}}\n\nPrevious feedback:\n{feedback}",
                agent=self.agents.general_physician(),
                expected_output="Revised answer (single choice) according to the feedback"
            )
        ]
        
    def output_task(self, task_description: str) -> Task:
        return Task(
            description=task_description,
            agent=self.agents.meta_agent(),
            expected_output="Final consensually agreed answer which is simply a single choice, nothing else"
        )