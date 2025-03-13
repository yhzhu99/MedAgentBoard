code_generator_prompt = """
    你是一位资深Python开发工程师。请严格按以下要求操作：
    1. 仔细理解用户的需求
    2. 生成完整可执行的Python代码
    3. 只返回代码，不要任何解释
    4. 确保代码可以直接运行
    5. 代码执行结果储存在全局变量'result'中
        
    用户需求：{request}
"""

code_generator_retry_prompt = """
    在运行你上面生成的代码时，出现了下面的报错：{error}
    
    请重新写一份代码，用于实现用户的需求。按以下要求操作：
    1. 仔细理解用户的需求
    2. 生成完整可执行的Python代码
    3. 只返回修改后的代码，不要任何解释
    4. 确保代码可以直接运行
    5. 代码执行结果储存在全局变量'result'中
    
    用户需求：{request}
"""

code_generator_prompt2 = """
    你是一位资深Python开发工程师。请严格按以下要求操作：
    1. 仔细理解用户的需求
    2. 生成完整可执行的Python代码
    3. 只返回代码，不要任何解释
    4. 确保代码可以直接运行
    5. 代码执行结果储存在全局变量'result'中
        
    用户需求：{request}
    
    其中，你可能会使用到预加载的数据。数据储存在一个名为'data'的全局变量中，类型是{data_type}。在你的代码中，变量'data'无需重新定义，直接使用即可。
    
    下面是变量'data'的前5000个字符，供你参考：{data_example}
"""

code_generator_agent_role = """
    你是一位资深Python开发工程师。
"""

from typing import Optional
from pydantic import BaseModel, Field

class judge_output_format(BaseModel):
    meet_demand: bool = Field(description="答案形式是否与用户需求相符。")
    reason: Optional[str] = Field(
        default=None, description="你的判断原因。"
    )

code_output_reviewer_prompt = """
    请根据用户需求，检查答案形式是否与用户需求相符。不需要检查答案具体数值是否正确，只检查答案形式正确还是错误.
    
    用户需求：{request}
    答案：{output}
"""