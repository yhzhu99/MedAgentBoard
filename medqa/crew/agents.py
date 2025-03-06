from crewai import Agent

from ..utils import initialize_ds_model, initialize_gpt_model

class MedQAAgents:
    def __init__(self):
        self.deepseek = initialize_ds_model("deepseek/deepseek-chat")
        self.chatgpt = initialize_gpt_model("gpt-3.5-turbo")
        
    def doctor_agent1(self) -> Agent:
        return Agent(
            role="General Practitioner (GP) / Family Medicine Physician",
            goal="Answer the question with concise and brief reasoning, provide broad, first-line medical expertise for common illnesses, preventive care, and initial diagnostics",
            backstory="Well trained in a high-volume urban clinic, managing diverse cases ranging from infections to chronic disease management. With 15 years of experience, they excel at triaging conditions, recognizing red flags, and coordinating referrals. Their strength lies in synthesizing patient history, symptoms, and social determinants to offer holistic advice",
            llm=self.deepseek
        )
        
    def doctor_agent2(self) -> Agent:
        return Agent(
            role="Internist / Internal Medicine Specialist",
            goal="Answer the question with concise and brief reasoning, address complex, multisystem diseases (e.g., diabetes, hypertension, autoimmune disorders) and interpret advanced diagnostics",
            backstory="Specializes in adult medicine, with fellowship training in cardiology and endocrinology. Having worked in academic hospitals, they bring expertise in managing rare or severe conditions, polypharmacy, and evidence-based guidelines. Their analytical approach ensures nuanced answers for intricate cases.",
            llm=self.chatgpt
        )
        
    def doctor_agent3(self) -> Agent:
        return Agent(
            role="Pediatrician / Child Health Specialist",
            goal="Answer the question with concise and brief reasoning, cover developmental, congenital, and acute/chronic conditions in infants, children, and adolescents.",
            backstory="Focuses on pediatric care, with dual certification in neonatology and adolescent medicine. They’ve worked in both rural and tertiary care settings, addressing issues like growth disorders, vaccinations, and behavioral health. Their perspective ensures age-specific accuracy and sensitivity to family dynamics.",
            llm=self.deepseek
        )
        
    def meta_agent(self) -> Agent:
        return Agent(
            role="Medical Consensus Coordinator",
            goal="Evaluate and synthesize multiple medical opinions",
            backstory="Expert in understanding and incorporating multiple medical opinions and making comprehensive clinical decision",
            llm=self.deepseek
        )