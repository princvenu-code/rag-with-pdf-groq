from langchain import hub
from langchain.agents import (AgentExecutor, create_vectorstore_agent, create_react_agent)
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

def get_current_time():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


tools = [Tool(name="Time",
              func=get_current_time,
              description="Useful to get the current time from system")]


prompt = hub.pull("hwchase17/react")

llm = ChatGroq(groq_api_key=groq_api_key,
               model="mixtral-8x7b-32768")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt, stop_sequence=True)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "WHat is the time now"})

print("Response Time Now: ", response)