from typing import Optional
from pydantic import BaseModel, Field

# Pydantic
class judge_output_format(BaseModel):

    need_code: str = Field(description="是否需要生成代码。只能返回一个单词：'Yes'或'No'，'Yes'代表需要生成代码，'No'代表不需要生成代码。")
    code_request: Optional[str] = Field(
        default=None, description="如果需要生成代码，将用户的代码需求转化为更精炼的形式。"
    )


structured_llm = llm.with_structured_output(judge_output_format)

structured_llm.invoke("Tell me a joke about cats")