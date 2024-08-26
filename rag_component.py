import constants as CONSTS
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
import logging
import uuid


def get_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders    
    logging.info(f"Loading documents from {source_dir}")
    paths = []
    for root, _, files in os.walk(source_dir):
        logging.info(f"Files list {files}")
        for file_name in files:
            logging.info(f"Importing {file_name}")
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in CONSTS.DOCUMENT_MAP.keys():                
                paths.append(source_file_path)
                logging.info(f"Appended file {file_name}")
    
    logging.info(f"Loading documents completed")
    return paths

def get_texts_from_documents(paths: list) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    logging.info(f"Reading texts from the documents")
    text = ""
    for path in paths:
        pdf_reader = PdfReader(path)
        for page in pdf_reader.pages:
            text+=page.extract_text()    
    logging.info(f"Reading texts completed")    
    return text

def extract_chuncks_from_text(text):
    logging.info(f"Creating text splits from text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splits = text_splitter.split_text(text)
    logging.info(f"Split into {len(text_splits)} chunks of text")
    return text_splits


def get_chuncks_from_docs(documents_directory):
    documents = get_documents(documents_directory)
    text = get_texts_from_documents(documents)
    text_splits = extract_chuncks_from_text(text=text)
    return text_splits

def get_documents_paths(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders    
    logging.info(f"Loading documents from {source_dir}")
    paths = []
    for root, _, files in os.walk(source_dir):
        logging.info(f"Files list {files}")
        for file_name in files:
            logging.info(f"Importing {file_name}")
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in CONSTS.DOCUMENT_MAP.keys():                
                paths.append(source_file_path)
                logging.info(f"Appended file {file_name}")    
    logging.info(f"Loading documents completed")
    return paths

def load_documents(documents_directory):
    logging.info(f"Loading documents started")
    loader = PyPDFLoader(documents_directory)
    documents = loader.load()
    logging.info(f"Loading documents completed")
    return documents

def load_single_document(document_path) -> Document:
    logging.info(f"Loading single document started")
    print(document_path)
    loader = PyPDFLoader(document_path)
    document = loader.load()
    logging.info(f"Loading single document completed")
    return document

def split_documents(document):
    logging.info(f"Splitting documents started")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splits = text_splitter.split_documents(documents=document)
    logging.info(f"Splitting documents completed")
    return text_splits

def get_text_splits_from_docs(documents_directory):
    documents = load_documents(documents_directory=documents_directory)
    text_splits = split_documents(documents=documents)
    return text_splits

def get_text_splits_from_single_doc(document_path):
    document = load_single_document(document_path=document_path)
    text_splits = split_documents(document=document)
    return text_splits

def add_documents_to_vector_store(documents_directory, embeddings):
    documents_paths = get_documents_paths(documents_directory)
    for path in documents_paths:
        add_doc_to_vector_store(path, embeddings)

def add_doc_to_vector_store(document_path, embeddings):
    text_splits = get_text_splits_from_single_doc(document_path)
    Chroma.from_documents(documents=text_splits, 
                          embedding=embeddings, 
                          persist_directory=CONSTS.PERSISTENT_DIRECTORY, 
                          client_settings=CONSTS.CHROMA_SETTINGS)    
    logging.info(f"Document added.")

def create_vector_store():
    if not os.path.exists(CONSTS.PERSISTENT_DIRECTORY):
        logging.info(f"Vector store doesn't exists")
        try:
            Chroma(persist_directory=CONSTS.PERSISTENT_DIRECTORY, client_settings=CONSTS.CHROMA_SETTINGS)
            logging.info(f"Vector store created successfully")
        except Exception as ex:
            logging.error(f"Failure in creating vector store. Exception: {str(ex)}")
    else:
        logging.info(f"Vector store already exists")

def get_vector_store():
    # if not os.path.exists(CONSTS.PERSISTENT_DIRECTORY):
    #     logging.info(f"Vector store doesn't exists")
    #     try:
    #         Chroma(persist_directory=CONSTS.PERSISTENT_DIRECTORY, client_settings=CONSTS.CHROMA_SETTINGS)
    #         logging.info(f"Vector store created successfully")
    #     except Exception as ex:
    #         logging.error(f"Failure in creating vector store. Exception: {str(ex)}")
    # else:
    #     logging.info(f"Vector store already exists")

    logging.info(f"Getting vector store")
    db = Chroma(persist_directory=CONSTS.PERSISTENT_DIRECTORY, client_settings=CONSTS.CHROMA_SETTINGS)
    logging.info(f"Returning vector store")
    return db

def get_vector_store_retriever(embeddings):
    db = Chroma(
        persist_directory=CONSTS.PERSISTENT_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CONSTS.CHROMA_SETTINGS
    )
    retriever =  db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 1, "score_threshold": 0.1})
    # retriever =  db.as_retriever()
    return retriever
