from langgraph.types import Command
from .crew.agents import CodeGeneratorAgent, CodeReviewerAgent
from .crew.tasks import CodeGenerationTask, CodeReviewTask
from .utils import extract_code_from_response
# from .crew.crew import CodeGenerationCrew, CodeReviewCrew

from crewai import Crew

class CodeNodes:
    def __init__(self):
        pass
        self.CodeGeneratorAgent = CodeGeneratorAgent()
        self.CodeReviewerAgent = CodeReviewerAgent()    # 是否每次运行至该节点都新建Agent？
    
    def code_generator(self, state):
        
        # 每次调用时使用同一个agent，不同的任务和Crew（prompt已在create_task中决定）
        task = CodeGenerationTask().create_task(self.CodeGeneratorAgent, state)  
        CodeGenerationCrew = Crew(
            agents=[task.agent], 
            tasks=[task],
        )
        
        result = CodeGenerationCrew.kickoff()
        clean_code = extract_code_from_response(result)
        
        return Command(
            update={
                "code": clean_code,
                "attempts": state["attempts"] + 1
            },
            # goto="code_executor",
        )
    
    def code_executor(self, state):
        code = state["code"]
        error = None
        output = None
        try:
            data = state['data']
            result = {}
            exec(code, {'data': data}, result)
            output = result['result']
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
        
        return Command(
            update={"error": error, 'code_output': output},
            # goto="code_reviewer"
        )
    
    def code_auto_reviewer(self, state):
        # 最大尝试次数设为3次
        MAX_ATTEMPTS = 3
        # 检查是否存在报错
        if state["error"] and state["attempts"] < MAX_ATTEMPTS:
            print(f"检测到错误，尝试重新生成（剩余尝试次数：{MAX_ATTEMPTS - state['attempts']}）")
            return Command(
                update = {},
                goto = "code_generator"
            )
        return Command(
            update = {},
        )   
    
    def code_reviewer(self, state):
        
        MAX_ATTEMPTS = 3
        
        task = CodeReviewTask().create_task(self.CodeReviewerAgent, state)  
        CodeReviewCrew = Crew(
            agents=[task.agent], 
            tasks=[task],
        )
        
        result = CodeReviewCrew.kickoff()
        
        if str(result) == 'no' and state["attempts"] < MAX_ATTEMPTS:
            return Command( # 这里有两个地方没有完成：1.如果回答yes或no之外的答案应该如何处理   2.返回Generator后应该如何修改prompt
                update = {},
                goto = "code_generator"
            )
            
        return Command(
            update={},
        )
        