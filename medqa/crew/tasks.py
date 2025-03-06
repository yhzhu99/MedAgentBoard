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
                agent=self.agents.doctor_agent1(),
                expected_output="As an Internist / Internal Medicine Specialist who has first-line medical expertise for common illnesses, preventive care, and initial diagnostics, Give one single answer and choice with brief reasoning and explanation for the choice"
            ),
            Task(
                description="{question}",
                agent=self.agents.doctor_agent2(),
                expected_output="As a Pediatrician / Child Health Specialist who has expertise in addressing complex, multisystem diseases (e.g., diabetes, hypertension, autoimmune disorders) and interpret advanced diagnostics. Give one single answer and choice with brief reasoning and explanation for the choice"
            ),
            Task(
                description="{question}",
                agent=self.agents.doctor_agent3(),
                expected_output="As a General Practitioner (GP) / Family Medicine Physiciancover with expertise on developmental, congenital, and acute/chronic conditions in infants, children, and adolescents. Give one single answer and choice with brief reasoning and explanation for the choice"
            )
        ]
    
    # Define the feedback task handled by the meta agent
    def feedback_task(self, state) -> Task:
        return Task(
            description="Analyze these answers and identify key disagreements:\n\n{'='*20}\n" +
                        "\n\n".join([f"Doctor {i+1}: {ans}" for i, ans in enumerate(state['answers'])]), # TODO:Rather than quoting each doctor as "Doctor i", use the role assigned to each doctor agent
            agent=self.agents.meta_agent(),
            expected_output="Summary of conflicting opinions and suggested focus areas for next round of collaboration and refinement to reach a consensus among the doctors"
        )
    
    # Define the refinement task handled by each doctor agent given the feedback
    def refinement_task(self, feedback: str) -> List[Task]:
        return [
            Task(
                description=f"{{question}}\n\nYou are doctor 1, here is the deedback from previous round of discussion with other doctors:\n{feedback}", # TODO: Add previous answers?
                agent=self.agents.doctor_agent1(),
                expected_output="Give one single revised answer/choice with brief reasoning and explanation according to the feedback"
            ),
            Task(
                description=f"{{question}}\n\nYou are doctor 2, here is the deedback from previous round of discussion with other doctors:\n{feedback}",
                agent=self.agents.doctor_agent2(),
                expected_output="Give one single revised answer/choice with brief reasoning and explanation according to the feedback"
            ),
            Task(
                description=f"{{question}}\n\nYou are doctor 3, here is the deedback from previous round of discussion with other doctors:\n{feedback}",
                agent=self.agents.doctor_agent3(),
                expected_output="Give one single revised answer/choice with brief reasoning and explanation according to the feedback of previous round"
            )
        ]
        
    def output_task(self, task_description: str) -> Task:
        return Task(
            description=task_description,
            agent=self.agents.meta_agent(),
            expected_output="Final consensually agreed answer which is simply a single choice, nothing else"
        )