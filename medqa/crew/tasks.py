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
                description='''
                Answer the question: 
                {question}, 
                and note the following instructions before providing the answer:
                [ROLE] You are a Primary Care Physician (PCP) , acting as the first point of contact for patients. Your role is to assess the question holistically, considering common conditions, preventive care, and the patient’s overall health history
                [Expertise] General medicine, chronic disease management (e.g., hypertension, diabetes), and preventive care. Strong focus on patient history, lifestyle factors, and mental health. Skilled in identifying red flags that require specialist referral or urgent care.
                ''',
                agent=self.agents.doctor_agent1(),
                expected_output= "Give one single answer and choice with brief reasoning and explanation (and why other options are wrong) as a Primary Care Physician (PCP). Begin your answer with 'As a Primary Care Physician (PCP), I would choose...'"
    
            ),
            Task(
                description='''
                Answer the question: 
                {question}, 
                and note the following instructions before providing the answer:
                [ROLE] You are a Specialist (e.g., Endocrinologist, Pathologist, Cardiologist, Neurologist). Your role is to analyze the question through the lens of your subspecialty, focusing on rare, complex, or systemic conditions. Provide deep technical insights and advanced diagnostic or therapeutic recommendations.
                [Expertise] Endocrinologist : Hormonal disorders (e.g., thyroid disease, diabetes complications). Pathologist : Lab results, biomarkers, and tissue analysis. Cardiologist : Heart and vascular system disorders. Neurologist : Neurological conditions (e.g., seizures, neuropathy). Other subspecialties as needed.
                ''',
                agent=self.agents.doctor_agent2(),
                expected_output="Give one single answer and choice with brief reasoning and explanation (and why other options are wrong) as a Specialist. Begin your answer with 'As a Specialist, I would choose...'"
            ),
            Task(
                description=
                '''
                Answer the question: 
                {question}, 
                and note the following instructions before providing the answer:
                [ROLE] You are a Surgeon , focusing on procedural and anatomical solutions. Your role is to evaluate whether the question involves a condition requiring surgical intervention, assess surgical risks/benefits, and consider alternatives like minimally invasive techniques.
                [Expertise] Surgical anatomy, pre/post-operative care, and complications. Skilled in evaluating trauma, tumors, structural defects, or emergencies (e.g., appendicitis, hernias). Knowledge of procedural alternatives (e.g., laparoscopy vs. open surgery)
                ''',
                agent=self.agents.doctor_agent3(),
                expected_output="Give one single answer and choice with brief reasoning and explanation (and why other options are wrong) as a Surgeon. Begin your answer with 'As a Surgeon, I would choose...'"
                
            )
        ]
    
    # Define the consensus check task handled by the meta agent
    def consensus_check_task(self, answers: List[str]) -> Task:
        return Task(
            description="Based on the answers from all doctors:" +
                        "\n\n".join([f"Doctor {i+1}: {ans}" for i, ans in enumerate(answers)]) +
                        "determine if a consensus is reached (same answer to the question) or not among the doctors",
            agent=self.agents.meta_agent(),
            expected_output="One-word output, nothing else: return 'True' if consensus is reached, return 'False' otherwise"
        )
    
    # Define the feedback task handled by the moderator agent
    def feedback_task(self, state) -> Task:
        return Task(
            description="Analyze these answers and identify key disagreements:\n\n{'='*20}\n" +
                        "\n\n".join([f"Doctor {i+1}: {ans}" for i, ans in enumerate(state['current_answers'])]), # TODO:Rather than quoting each doctor as "Doctor i", use the role assigned to each doctor agent
            agent=self.agents.moderator_agent(),
            expected_output="Summary of conflicting opinions among the doctors"
        )
    
    # Define the refinement task handled by each doctor agent given the feedback
    def refinement_task(self, question: str, previous_answers: list, feedback: str) -> List[Task]:
        return [
            Task(
                description=f"Answer the question:\n{question}\nYou are doctor 1, your answer and reasoning to the question in the previous round is:\n{previous_answers[0]}\nAnd here is the feedback from previous round of discussion with other doctors:\n{feedback}",
                agent=self.agents.doctor_agent1(),
                expected_output="Give one single revised answer/choice with brief reasoning and explanation according to the feedback, your previous answer and your role and expertise"
            ),
            Task(
                description=f"Answer the question:\n{question}\nYou are doctor 2, your answer and reasoning to the question in the previous round is:\n{previous_answers[1]}\nAnd here is the feedback from previous round of discussion with other doctors:\n{feedback}",
                agent=self.agents.doctor_agent2(),
                expected_output="Give one single revised answer/choice with brief reasoning and explanation according to the feedback, your previous answer and your role and expertise"
            ),
            Task(
                description=f"Answer the question:\n{question}\nYou are doctor 3, your answer and reasoning to the question in the previous round is:\n{previous_answers[2]}\nAnd here is the feedback from previous round of discussion with other doctors:\n{feedback}",
                agent=self.agents.doctor_agent3(),
                expected_output="Give one single revised answer/choice with brief reasoning and explanation according to the feedback, your previous answer and your role and expertise"
            )
        ]
        
    def output_task(self, task_description: str) -> Task:
        return Task(
            description=task_description,
            agent=self.agents.moderator_agent(),
            expected_output="Final consensually agreed answer which is simply a single choice, nothing else, e.g., 'A'"
        )