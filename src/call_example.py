import os
from crewai import Agent, Task, Crew
from tool.call import CodeTool, CodeTool2

# 设置 API 密钥
os.environ["OPENAI_API_KEY"] = "sk-2Cd64920f5df1880606c597a75f9f05009d2d9222a4Jur6C"
os.environ["OPENAI_API_BASE"] = "https://api.gptsapi.net/v1" # OpenAI API 基础地址

# 创建代理
agent1 = Agent(
    role='研究员',
    goal='解决用户提出的问题',
    backstory='一位擅长运用代码工具解决问题的研究员。',
    tools=[CodeTool2(), ],
    verbose=True
)

# 定义任务
agent1_task = Task(
    description='解决用户提出的问题：将数据中提供的数字按从大到小排列，其中数据的地址为"/home/kisara/RESEARCH/test.json"。',
    expected_output='排列后的数字。',
    agent=agent1
)

# 组建一个启用规划的团队
crew = Crew(
    agents=[agent1],
    tasks=[agent1_task],
    verbose=True,
    planning=True,  # 启用规划功能
)

# 执行任务
crew.kickoff()

    