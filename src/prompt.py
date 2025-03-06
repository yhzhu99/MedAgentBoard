from typing import Optional
from pydantic import BaseModel, Field

# prompt内容
judge_prompt = """
    请严格按照下面的要求操作：
    1.阅读用户需求，判断是否需要生成一段python代码用于辅助解决问题；
    2.返回一个单词：'Yes'或'No'，'Yes'代表需要生成代码，'No'代表不需要生成代码。
    3.如果需要生成代码，将用户的代码需求转化为更精炼的形式。
    
    用户需求：{request}
    其中，用户提供了一份EHR数据，数据的前5000个字符如下：{data}
"""

# prompt内容
judge_prompt_2 = """
    请严格按照下面的要求操作：
    1.阅读用户需求，判断是否需要生成一段python代码用于辅助解决问题；
    2.如果需要生成代码，将用户的代码需求转化为更精炼的形式。
    
    用户需求：{request}
    其中，用户提供了一份EHR数据，数据的前5000个字符如下：{data}
"""
    
    
# 输出格式限制
class judge_output_format(BaseModel):
    
    # name: Optional[str] = Field(default=None)  # 添加 name 字段?
    
    need_code: str = Field(description="是否需要生成代码。只能返回一个单词：'Yes'或'No'，'Yes'代表需要生成代码，'No'代表不需要生成代码。")
    code_request: Optional[str] = Field(
        default=None, description="如果需要生成代码，将用户的代码需求转化为更精炼的形式。"
    )


system_prompt = """
    请以json格式返回答案。
"""