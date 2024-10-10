import os
from apikey import apikey

from langchain import hub
from langchain_openai import OpenAI
from langchain.agents import load_tools, create_react_agent, AgentExecutor

os.environ["OPENAI_API_KEY"] = apikey

llm = OpenAI(temperature=0.0)

tools = load_tools(['wikipedia', 'llm-math'], llm)
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm,tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

answer = input('Input Wikipedia Research Task\n')

agent_executor.invoke({'input':answer})