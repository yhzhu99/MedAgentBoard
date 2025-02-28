# How to use @tool

Documentation: https://python.langchain.com/docs/concepts/tools/

## How to store `local variables` in the `tool` function

https://python.langchain.com/docs/how_to/tool_artifacts/

Background: There are `artifacts` (which means intermediate or local variables) of a tool's execution that we want to make accessible to downstream components in our chain or agent, but that we don't want to expose to the model itself.

Solution: Use `@tool(response_format="content_and_artifact")` and `return content, artifact`

Careful: If we just pass in the tool call args, we'll only get back the content. We need to invoke with `AIMessage.tool_calls`


## How to pass the graph state into the `tool`

https://python.langchain.com/docs/how_to/tool_runtime/

Solution: use `from langchain_core.tools import InjectedToolArg`

## How to update the graph state when excuting the `tool`

https://github.com/langchain-ai/langgraph/discussions/1616

### Easy example: 

Imagine we have several tool (python function object) for taking `query` as input and generating answer in different domains.

```python
from langchain_core.tools.base import InjectedToolArg
from langchain_core.tools import BaseTool
from typing import Any

@tool
def search_tool_1(any: Any, query: Annotated[str, InjectedToolArg]):
    """A tool that can get information related to robotics.
    
    Args:
        any: The bar.
        query: https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html
    """
    ans = TOOL_1(query)
    return Command(
        update = {
                "ans": ans,
                },
        goto="reviewer_agent"
    )

@tool
def search_tool_2(any: Any, query: Annotated[str, InjectedToolArg]):
    """A tool that can get information related to large language model.

    Args:
        any: The bar.
        query: The baz. https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html
    """
    ans = TOOL_2(query)
    return Command(
        update = {
                "ans": ans,
                },
        goto="reviewer_agent"
    )

# If we run `search_tool_2.tool_call_schema.schema()`, we will get the schema of the tool call, which to my understanding, is LLM generated input schema -> the AIMessage.tool_calls: list[dict]. 
# So here LLM only needs to generate arg: any!
# {'description': 'A tool that can get information related to large language model.',
#  'properties': {'any': {'title': 'Any'}},
#  'required': ['any'],
#  'title': 'search_tool_2',
#  'type': 'object'}

# If we run `search_tool_2.get_input_schema().schema()`
# {'description': 'A tool that can get information related to large language model.',
#  'properties': {'any': {'title': 'Any'},
#   'query': {'title': 'Query', 'type': 'string'}},
#  'required': ['any', 'query'],
#  'title': 'search_tool_2',
#  'type': 'object'}

def search_node(state: MessagesState):
    agent_instruction = HumanMessage(content="""You need to find a tool to retrieve information based on queries \n ## query \n state["query"]""")

    tools = [search_tool_1, search_tool_2]
    tools_by_name: dict[str, BaseTool] = {_tool.name: _tool for _tool in tools}

    # Tool binding
    llm_with_tools = llm.bind_tools(tools)
    # get the response: AIMessage from the llm_with_tools.invoke
    response = llm_with_tools.invoke([agent_instruction])

    # explanation: in response.tool_calls, we have the tool name and the arguments excluding the InjectedToolArg
    # we need to invoke the tool with the InjectedToolArg
    commands = [tools_by_name[tool_call["name"]].invoke({**tool_call["args"], "query":state["query"]}) for tool_call in response.tool_calls]
    
    return commands # return List[Command]
```

**So in other words, what we do here is to wrap a customized function using the `@tool` decorator, to bind the tools with LLM, to generate tool calling (AIMessage) with LLM,  and call the function.**
