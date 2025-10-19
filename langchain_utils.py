from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from chroma_utils import vectorstore
from config import api_key, api_version, api_endpoint, api_model, langsmith_key

load_dotenv()

# langsmith_key = os.getenv("LANGCHAIN_API_KEY")

# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=langsmith_key
# os.environ["LANGCHAIN_PROJECT"]="rag-fastapi-project"


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

output_parser = StrOutputParser()

#Setting Up Prompts
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),         
    ("human", "{input}")
])
##The chat history helps maintain conversational continuity.##

# Creating the RAG Chain
def get_rag_chain(model=os.environ["OPENAI_AZURE_MODEL"]):
    #llm = ChatOpenAI(model=model)
    llm=AzureChatOpenAI(
        api_key= api_key,
        api_version= api_version,
        azure_endpoint= api_endpoint,
        model_name= api_model,
        temperature=0.4,
        max_tokens=1000,
        seed=42,)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain
