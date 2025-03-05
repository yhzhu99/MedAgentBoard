from crewai import Agent

from ..utils import initialize_ds_model

class MedQAAgents:
    def __init__(self):
        self.model_name = "deepseek/deepseek-chat"
        self.llm = initialize_ds_model(self.model_name)
        
    def cardiologist(self) -> Agent:
        return Agent(
            role="Cardiologist",
            goal="Answer the question with in-depth and step-by-step reasoning with your expertise as a cardiologist",
            backstory="Board-certified cardiologist with 15 years experience",
            llm=self.llm
        )
        
    def neurologist(self) -> Agent:
        return Agent(
            role="Neurologist",
            goal="Answer the question with concise and brief reasoning with your expertise as a neurologist",
            backstory="Leading neurology expert at major university hospital",
            llm=self.llm
        )
        
    def general_physician(self) -> Agent:
        return Agent(
            role="General Physician",
            goal="Answer the question with concise and brief reasoning with your expertise as a general physician",
            backstory="Experienced GP with holistic approach to diagnosis",
            llm=self.llm
        )
        
    def meta_agent(self) -> Agent:
        return Agent(
            role="Medical Consensus Coordinator",
            goal="Evaluate and synthesize multiple medical opinions",
            backstory="Expert in understanding and incorporating multiple medical opinions and making comprehensive clinical decision",
            llm=self.llm
        )