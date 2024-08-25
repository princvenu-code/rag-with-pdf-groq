from langchain import hub
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain.tools import Tool
from langchain.agents import (AgentExecutor, create_react_agent, create_structured_chat_agent, create_vectorstore_agent)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

cur_dir = os.path.dirname(os.path.abspath(__file__))

db_dir = os.path.join(cur_dir, "..", "..", "4_rag", "db")
pers_dir = os.path.join(db_dir, "Chroma_db_with+metadata")


if os.path.exists(pers_dir):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=pers_dir, embedding_function=None)
else:
    raise FileExistsError(
        f"The directory doesn't exit. Please check the path"
    )

embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )


