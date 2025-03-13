from crewai import Task
from .prompts import *

class CodeGenerationTask:   # agent是如何实现多轮对话的？
    def __init__(self):
        pass
    
    def create_task(self, CodeGeneratorAgent, state):
        if state['attempts'] == 0:
            prompt = code_generator_prompt2.format(
                request=state['request'],
                data_type=str(type(state['data'])),
                data_example=str(state['data'])[:5000]
            ) if state['data'] else code_generator_prompt.format(
                request=state['request']
            )
        else:
            prompt = code_generator_retry_prompt.format(
                error=state['error'],
                request=state['request']
            )
        
        # print('\n\n\n\n')
        # print('Task description:', prompt)
        # print('Task expected_output:', "Valid Python code block enclosed in triple backticks")
        # print('Task agent:', CodeGeneratorAgent.agent)
        
        return Task(
            description=prompt,
            expected_output="Valid Python code block enclosed in triple backticks",
            agent=CodeGeneratorAgent.agent
        )
        
class CodeReviewTask:
    def __init__(self):
        pass
    
    def create_task(self, CodeReviewerAgent, state):
        return Task(
            description=code_output_reviewer_prompt.format(
                request=state['request'],
                output=state['code_output']
            ),
            # expected_output="Structured JSON validation result", 
            expected_output="A single word 'yes' or 'no'.", # 这里先简单使用单个单词
            agent=CodeReviewerAgent.agent
        )