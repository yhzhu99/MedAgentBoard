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
                Answer the {question} and note the following instructions before providing the answer:
                [ROLE] Senior Family Physician with 15 years urban clinic experience
                [FOCUS] Prioritize common conditions (80% prevalence) before considering rare diagnoses. Always assess for red flags requiring emergency care. Consider practical constraints (limited testing availability, patient compliance). Highlight cost-effective first-line interventions and preventive measures.
                [REASONING FRAMEWORK]
                1. Start with most statistically likely diagnosis based on prevalence
                2. Evaluate symptom patterns against WHO primary care guidelines
                3. Assess social determinants (housing, income, health literacy)
                4. Identify urgent referral thresholds using NICE criteria
                5. Recommend stepwise management: Lifestyle > OTC meds > Prescriptions
                ''',
                agent=self.agents.doctor_agent1(),
                expected_output= "Give one single answer and choice with brief reasoning and explanation as a Senior Family Physician with 15 years urban clinic experience."
    
            ),
            Task(
                description='''
                Answer the {question} and note the following instructions before providing the answer:
                [ROLE] Academic Hospital Internist specializing in complex multimorbidity
                [FOCUS] Identify atypical presentations of systemic diseases. Analyze potential drug interactions. Consider secondary/tertiary prevention strategies. Evaluate diagnostic uncertainty through Bayesian analysis.
                [REASONING FRAMEWORK]
                1. Generate differentials using the "VINDICATE" framework (Vascular, Infectious, etc.)
                2. Apply diagnostic test interpretation principles (sensitivity/Specificity, LR ratios)
                3. Assess organ system interactions using pathophysiological models
                4. Reference latest NEJM/BMJ guidelines for specialist management
                5. Weigh risks/benefits of advanced imaging or invasive testing
                ''',
                agent=self.agents.doctor_agent2(),
                expected_output="Give one single answer and choice with brief reasoning and explanation as a Academic Hospital Internist specializing in complex multimorbidity."
            ),
            Task(
                description=
                '''
                Answer the {question} and note the following instructions before providing the answer:
                [ROLE] Board-Certified Pediatrician with Neonatology/Adolescent Medicine Expertise  
                [FOCUS] Prioritize developmental appropriateness in all assessments. Consider growth patterns, vaccine status, and family/caregiver capabilities. Distinguish between normal developmental variations and pathological findings.
                [REASONING FRAMEWORK]  
                1. **Age Stratification**: Immediately establish patient's Tanner stage/developmental phase  
                2. **Growth Analysis**: Compare to WHO growth charts - flag <-2SD in height/weight/BMI  
                3. **Vaccine Cross-Check**: Verify against CDC schedule for age + catch-up needs  
                4. **Developmental Red Flags**: Screen for missed milestones using ASQ-3 domains  
                5. **Family Dynamics**: Apply HEADSSS assessment (Home, Education, etc.) for adolescents 
                ''',
                agent=self.agents.doctor_agent3(),
                expected_output="Give one single answer and choice with brief reasoning and explanation as a Board-Certified Pediatrician with Neonatology/Adolescent Medicine Expertise ."
                
            )
        ]
    
    # Define the consensus check task handled by the meta agent
    def consensus_check_task(self, answers: List[str]) -> Task:
        return Task(
            description="Based on the answers from all doctors:\n\n{'='*20}\n" +
                        "\n\n".join([f"Doctor {i+1}: {ans}" for i, ans in enumerate(answers)]) +
                        "determine if a consensus is reached (same answer to the question) or not among the doctors",
            agent=self.agents.meta_agent(),
            expected_output="One-word output, nothing else: return 'True' if consensus is reached, return 'False' otherwise"
        )
    
    # Define the feedback task handled by the moderator agent
    def feedback_task(self, state) -> Task:
        return Task(
            description="Analyze these answers and identify key disagreements:\n\n{'='*20}\n" +
                        "\n\n".join([f"Doctor {i+1}: {ans}" for i, ans in enumerate(state['answers'])]), # TODO:Rather than quoting each doctor as "Doctor i", use the role assigned to each doctor agent
            agent=self.agents.moderator_agent(),
            expected_output="Summary of conflicting opinions among the doctors"
        )
    
    # Define the refinement task handled by each doctor agent given the feedback
    def refinement_task(self, feedback: str) -> List[Task]:
        return [
            Task(
                description=f"{{question}}\n\nYou are doctor 1, here is the feedback from previous round of discussion with other doctors:\n{feedback}", # TODO: Add previous answers?
                agent=self.agents.doctor_agent1(),
                expected_output="Give one single revised answer/choice with brief reasoning and explanation according to the feedback"
            ),
            Task(
                description=f"{{question}}\n\nYou are doctor 2, here is the feedback from previous round of discussion with other doctors:\n{feedback}",
                agent=self.agents.doctor_agent2(),
                expected_output="Give one single revised answer/choice with brief reasoning and explanation according to the feedback"
            ),
            Task(
                description=f"{{question}}\n\nYou are doctor 3, here is the feedback from previous round of discussion with other doctors:\n{feedback}",
                agent=self.agents.doctor_agent3(),
                expected_output="Give one single revised answer/choice with brief reasoning and explanation according to the feedback of previous round"
            )
        ]
        
    def output_task(self, task_description: str) -> Task:
        return Task(
            description=task_description,
            agent=self.agents.moderator_agent(),
            expected_output="Final consensually agreed answer which is simply a single choice, nothing else, e.g., 'A'"
        )