from crewai import Agent
from tool.utils import initialize_gpt_model # 引用方式好像有问题

class CodeGeneratorAgent:
    def __init__(self):
        self.llm = initialize_gpt_model("gpt-4o")
        self.agent = Agent(
            role="Senior Python Developer",
            goal="Generate executable Python code based on requirements",
            backstory="""Experienced in creating efficient and correct Python code.
                Specializes in data processing and analysis tasks.""",
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    

class CodeReviewerAgent:
    def __init__(self, ):
        self.llm = initialize_gpt_model("gpt-4o")
        self.agent = Agent(
            role="Answer Assurance",
            goal="Ensure answer format correctness",
            backstory="""Expert skilled in identifying logical errors.""",
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )