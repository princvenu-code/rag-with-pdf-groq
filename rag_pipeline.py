from rag_component import *
import constants as CONSTS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains import retrieval_qa, ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import logging


class RAGPipeline:
    def __init__(self, ingest_data: bool):
            self.ingest_data = ingest_data
            load_dotenv()
            logging.basicConfig(
                    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", 
                    level=logging.INFO)
            
            groq_api_key = os.environ["GROQ_API_KEY"]

            self.llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name='mixtral-8x7b-32768'
                )

            # embeddings = HuggingFaceEmbeddings(
            #         model_name="thenlper/gte-small",
            #         multi_process=False,
            #         model_kwargs={"device": "cpu"},
            #         encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
            #     )
            self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
            if self.ingest_data:
                add_documents_to_vector_store(CONSTS.SOURCE_DIRECTORY, self.embeddings)

            self.retriever = get_vector_store_retriever(embeddings=self.embeddings)

    def get_conversation_chain(self):
        message_history = ChatMessageHistory()    
           
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )


        chain = ConversationalRetrievalChain.from_llm(
            llm = self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            memory=memory,
            return_source_documents=True,
        )
        return chain


    def get_raw_rag_response(self, query):
         
        response = self.retriever.invoke(query)
        print("----Response----")

        for i, doc in enumerate(response, 1):
            print(f"Document {i}: \n{doc.page_content}\n")

    
    def get_response_from_chain(self, chain, query):
        res = chain.invoke({"question": query})
        answer = res["answer"]
        source_documents = res["source_documents"] 
        return answer
    

    def main(self):
        chain = self.get_conversation_chain()
        query = "What are different types of windowing?"
        response = self.get_response_from_chain(chain, query)
        # response = self.get_raw_rag_response(query)
        print(response)

if __name__ == '__main__':
    try:
        rag_pipeline = RAGPipeline(False)        
        rag_pipeline.main()
        logging.info(f">>>>>> RAG Pipleline Started <<<<<<\n\nx==========x")
    except Exception as e:
        logging.info(f"Exception ", str(e))
        raise e
