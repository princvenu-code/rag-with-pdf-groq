from langchain import hub
from rag_pipeline import RAGPipeline
from langchain.agents import (AgentExecutor, create_vectorstore_agent, create_react_agent, create_structured_chat_agent)
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

def get_current_time():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def get_current_date():
    import datetime
    now = datetime.datetime.now()
    return now.date("%DD:$MM:%YYYY %p")


def search_wikipedia(query):
    from wikipedia import summary

    try:
        return summary(query, sentences=5)
    except:
        "I couldn't find the information on that"

def search_docstore(query):
    rag_pipeline = RAGPipeline(False)
    chain = rag_pipeline.get_conversation_chain()
    try:
        return rag_pipeline.get_response_from_chain(chain=chain, query=query)
    except:
        "I couldn't find the information on that"



tools = [Tool(name="Time",
              func=get_current_time,
              description="Useful to get the current time from system"
              ),
        Tool(name="Date",
              func=get_current_date,
              description="Useful to get the current date from system"
              ),
        Tool(name="wikipedia",
             func=search_wikipedia,
             description="Useful for when you need to know information about the topic"
             ),
        Tool(name="docstore",
             func=search_docstore,
             description="Useful for when you need to answer question about the context from the document"
             ),
             ]


prompt = hub.pull("hwchase17/structured-chat-agent")

llm = ChatGroq(groq_api_key=groq_api_key,
               model="mixtral-8x7b-32768")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_structured_chat_agent(
    llm=llm, 
    tools=tools, 
    prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True,
    memory=memory,
    handle_parsing_errors=True)

initial_message = '''
You are an AI assistant that can provide helpful answers using available tools. \nIf you get any context from the documents, then answer only that. If you are not able to get any 
context from documents, please search the wikipedia for the information
''' 
memory.chat_memory.add_message(AIMessage(content=initial_message))
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break


    memory.chat_memory.add_message(HumanMessage(content=user_input))

    response = agent_executor.invoke({"input": user_input})
    print("Bot: ", response["output"])

    memory.chat_memory.add_message(AIMessage(content=response["output"]))