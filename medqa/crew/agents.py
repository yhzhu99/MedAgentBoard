from crewai import Agent

from medqa.utils import *

class MedQAAgents:
    def __init__(self):
        self.deepseek = initialize_ds_model("deepseek/deepseek-chat")
        self.qwen_plus = initialize_qwen_model("openai/qwen-plus-latest")
        self.qwen_max = initialize_qwen_model("openai/qwen-max-latest")
        
        
    def doctor_agent1(self) -> Agent:
        return Agent(
            role="General Practitioner",
            goal="Answer the question with concise and brief reasoning, assess the question holistically, considering common conditions, preventive care, and the patient’s overall health history.",
            backstory="A very experienced general practitioner who have seen a decent amount of cases and patients.",
            llm=self.deepseek
        )
        
    def doctor_agent2(self) -> Agent:
        return Agent(
            role="Medical Specialist (e.g., Endocrinologist, Pathologist, Cardiologist, Neurologist)",
            goal="Answer the question with concise and brief reasoning, analyze the question through the lens of your subspecialty, focusing on rare, complex, or systemic conditions.",
            backstory="A very experienced medical specialist who have seen a decent amount of cases and patients.",
            llm=self.deepseek
        )
        
    def doctor_agent3(self) -> Agent:
        return Agent(
            role="Surgeon",
            goal="Answer the question with concise and brief reasoning, evaluate whether the question involves a condition requiring surgical intervention, assess surgical risks/benefits, and consider alternatives like minimally invasive techniques.",
            backstory="A very experienced surgeon who have seen a decent amount of cases and patients.",
            llm=self.deepseek
        )
        
    def moderator_agent(self) -> Agent:
        return Agent(
            role="Medical Discussion Moderator",
            goal="Understand the answers from all doctors: If disagreements among doctors exist, identify key disagreements; If consensus is reached, summarize the agreed-upon answer and output the final brief answer",
            backstory="A experienced moderator with a background in medical ethics and conflict resolution. They have facilitated numerous interdisciplinary rounds, ensuring respectful communication and shared decision-making. Their role is to synthesize diverse perspectives, clarify misunderstandings, and guide the team towards a unified conclusion.",
            llm=self.deepseek
        )
    
    def meta_agent(self) -> Agent:
        return Agent(
            role="Medical Consensus Coordinator",
            goal="Based on the answers from all doctors, determine if a consensus is reached (same answer to the question) or not. Return 'True' if consensus is reached, 'False' otherwise",
            backstory="A experienced coordinator who can differentiate between consensus and disagreement. They have a background in medical ethics and conflict resolution.",
            llm=self.deepseek
        )