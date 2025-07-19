from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
import streamlit as st

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o", temperature = 0) # I want to minimize hallucination - temperature = 0 makes the model output more deterministic 


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},  # or "cuda" if you have GPU
    encode_kwargs={"normalize_embeddings": True}  # Recommended for BGE
)


#Pinecone setup
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
PINECONE_CLOUD = st.secrets["PINECONE_CLOUD"]
PINECONE_REGION = st.secrets["PINECONE_REGION"]

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )

# PDF loader and chunking logic as a function for reuse

def process_pdf_to_pinecone(pdf_path: str, index_name: str = PINECONE_INDEX_NAME):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    pdf_loader = PyPDFLoader(pdf_path)
    try:
        pages = pdf_loader.load()
        print(f"PDF has been loaded and has {len(pages)} pages")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    pages_split = text_splitter.split_documents(pages)
    # Store in Pinecone
    vectorstore = LangchainPinecone.from_documents(
        documents=pages_split,
        embedding=embedding_model,
        index_name=index_name,
        namespace="default"
    )
    print(f"Uploaded {len(pages_split)} chunks to Pinecone index '{index_name}'!")
    return vectorstore

# Get retriever from Pinecone index

def get_pinecone_retriever(index_name: str = PINECONE_INDEX_NAME):
    vectorstore = LangchainPinecone(
        index_name=index_name,
        embedding=embedding_model,
        namespace="default"
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    return retriever

# Tool for retrieval
@tool
def retriever_tool(query: str) -> str:
    """This tool retrieves relevant chunks from the vector store."""
    retriever = get_pinecone_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the document."
    results = []
    for i, doc in enumerate(docs):
        results.append(f'Chunk {i+1}:\n{doc.page_content}') # Here we format the output to include the chunk number and its content

    return "\n\n".join(results)

tools = [retriever_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools


# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls #grab the last message's tool calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}") # E.g Calling Tool: retriever_tool with query: stock market performance 2024
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else: # Calls the tool with the provided arguments from the LLM
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}") # length of the result
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

# If running as script, process the default PDF
if __name__ == "__main__":
    #pdf_path = "D:/Mussab Work/Lang Graph/data/Stock_Market_Performance_2024.pdf"
    #process_pdf_to_pinecone(pdf_path)
    running_agent()