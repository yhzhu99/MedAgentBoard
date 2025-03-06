from openai import OpenAI
import json
from typing import Dict, Callable, List, Optional

api_key="sk-2Cd64920f5df1880606c597a75f9f05009d2d9222a4Jur6C"
base_url="https://api.gptsapi.net/v1"

class Agent:
    def __init__(self, model: str = "gpt-4o-2024-08-06"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.messages: List[Dict] = []
        self.tools: List[Dict] = []
        self.tool_functions: Dict[str, Callable] = {}
        
        # 初始化结构化输出工具
        self._init_structured_output_tool()
        # 添加系统消息
        self._add_system_message()
    
    def _init_structured_output_tool(self):
        """初始化结构化输出工具"""
        self.structured_output_tool = {
            "type": "function",
            "function": {
                "name": "structured_output",
                "description": "Format the final response with structured data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response": {"type": "string", "description": "回答内容"},
                        "category": {"type": "string", "description": "回答分类"},
                        "confidence": {"type": "number", "description": "置信度"}
                    },
                    "required": ["response"]
                }
            }
        }
        self.register_tool(
            self.structured_output_tool,
            lambda **kwargs: kwargs  # 直接返回参数作为结构化结果
        )
    
    def _add_system_message(self):
        """添加系统提示"""
        system_msg = {
            "role": "system",
            "content": (
                "你是一个智能助手。请遵守以下规则：\n"
                "1. 需要调用工具时必须使用工具调用\n"
                "2. 最终回答必须使用structured_output工具返回结构化数据\n"
                "3. 保持对话自然流畅"
            )
        }
        self.messages.append(system_msg)
    
    def register_tool(self, tool_schema: Dict, tool_function: Callable):
        """注册工具
        
        Args:
            tool_schema: 工具模式，符合OpenAI工具定义规范
            tool_function: 工具对应的执行函数
        """
        self.tools.append(tool_schema)
        self.tool_functions[tool_schema["function"]["name"]] = tool_function
    
    def _process_tool_call(self, tool_call) -> Dict:
        """处理单个工具调用"""
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            # if function_name not in self.tool_functions:
            #     return {
            #         "role": "tool",
            #         "content": json.dumps({"error": f"工具{function_name}未注册"}),
            #         "tool_call_id": tool_call.id
            #     }
            
            result = self.tool_functions[function_name](**arguments)
            return {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            }
        except json.JSONDecodeError:
            return {
                "role": "tool",
                "content": json.dumps({"error": "参数解析失败"}),
                "tool_call_id": tool_call.id
            }
        except Exception as e:
            return {
                "role": "tool",
                "content": json.dumps({"error": str(e)}),
                "tool_call_id": tool_call.id
            }
    
    def chat(self, user_input: str) -> Dict:
        """执行对话轮次
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            Dict: 结构化输出结果
        """
        self.messages.append({"role": "user", "content": user_input})
        
        while True:
            # 调用OpenAI接口
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.5
            )
            
            message = response.choices[0].message
            self.messages.append(message.to_dict())
            
            # 没有工具调用时直接返回
            if not message.tool_calls:
                return {"response": message.content}
            
            # 处理工具调用
            final_output = None
            for tool_call in message.tool_calls:
                tool_response = self._process_tool_call(tool_call)
                self.messages.append(tool_response)
                
                # 检查是否是结构化输出工具（这里是利用了一个函数，强制llm返回，很不好。）
                if tool_call.function.name == "structured_output":
                    try:
                        final_output = json.loads(tool_response["content"])
                    except:
                        final_output = {"response": "输出解析失败"}
            
            # 如果获得最终输出则返回
            if final_output:
                return final_output
    
    def reset(self):
        """重置对话历史"""
        self.messages = [self.messages[0]]  # 保留系统消息
        
if __name__=='__main__':
    # 初始化Agent
    agent = Agent()

    # 注册自定义工具（示例：天气查询）
    def get_weather(location: str):
        """模拟天气查询工具"""
        return {
            "weather": "晴",
            "temperature": 25,
            "humidity": 60
        }

    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地区的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如：北京"
                    }
                },
                "required": ["location"]
            }
        }
    }

    agent.register_tool(weather_tool, get_weather)

    # # 无tool对话
    # response = agent.chat("写一首七言绝句，解释量子力学的定义。")
    
    # 示例对话
    response = agent.chat("北京今天天气怎么样？")
    print("第一次响应:", response)

    # response = agent.chat("那上海呢？")
    # print("第二次响应:", response)
    
    print('\n\n')
    print(agent.messages)

    # 输出示例：
    # 第一次响应: {'response': '北京今天天气晴，气温25度', 'category': 'weather', 'confidence': 0.95}
    # 第二次响应: {'response': '上海今天天气晴，气温27度', 'category': 'weather', 'confidence': 0.93}