from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.agents import create_react_agent
from langchain.tools import Tool
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
import time
import json
import os
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# import chromadb
# import pandas as pd
# from langchain.document_loaders import DataFrameLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain import HuggingFaceHub
# from langchain.llms import LlamaCpp


# === Layout & Style ===
st.set_page_config(page_title="Chat", layout="wide")

# === Sidebar ===
with st.sidebar:
    st.title("💬 Chat App")
    if st.button("🆕 Reset Chat History"):
        st.session_state['messages'] = []  # Reset conversation

    st.markdown("---")
    # st.subheader("📂 Chats")  # Static for now
    # st.write("Chat About Aftab")
    # st.selectbox("Select chat", options=["Chat 1", "Chat 2"], index=0)  # Placeholder

# === Initialize Session State ===
if "messages" not in st.session_state:
    st.session_state.messages = []



# === Groq LLM ===
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama3-70b-8192",
    temperature=0.7
)
# Use Open-Source Embeddings for Retrieval
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Persist ChromaDB for Vector Search
directory_cdb = './chromadb'

if os.path.exists(directory_cdb) and os.listdir(directory_cdb):
    chroma_db = Chroma(persist_directory=directory_cdb, embedding_function=embedding_model)
else:
    # Load your personal documents/texts for the first time only
    from langchain.schema import Document
    with open('./personal_info.json', 'r') as file:
        json_data = json.load(file)
    # Ensure it's a list of dicts with "page_content"
    documents = []
    for item in json_data:
        # Each item can be a string or dict with extra metadata
        if isinstance(item, str):
            documents.append(Document(page_content=item))
        elif isinstance(item, dict) and "page_content" in item:
            documents.append(Document(page_content=item["page_content"], metadata=item.get("metadata", {})))
        else:
            raise ValueError("Invalid format in personal_info.json. Each item must be a string or a dict with 'page_content'.")

    chroma_db = Chroma.from_documents(documents, embedding_model, persist_directory=directory_cdb)
    chroma_db.persist()

# Configure Memory for Conversation
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=10,  # Store last 4 messages
    return_messages=True
)
#  Build RAG Pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chroma_db.as_retriever()
)
# Define a Medical Knowledge Tool
tools = [
    Tool(
        name="Personal Information KB",
        func=qa.invoke,
        description="Use this tool to provide information about Aftab"
    )
]
#  Pull ReAct Prompt from LangChain Hub
prompt = hub.pull("hwchase17/react-chat")
#  Create a ReAct Agent
agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=prompt
)
#  Create an Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=conversational_memory,
    max_iterations=30,
    max_execution_time=600,
    handle_parsing_errors=True
)



# === Main UI ===
st.markdown("<h1 style='text-align: left;'>Chats</h1>", unsafe_allow_html=True)
user_input = st.chat_input("Type your message...")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process user message
if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Show bot thinking spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chat_history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user"
                    else AIMessage(content=m["content"])
                    for m in st.session_state.messages
                ]
                refined_input = {"input": f"{user_input}"}
                response = agent_executor.invoke(refined_input)
                assistant_reply = response.output

                # Optional: Simulate typing effect
                # Simulate a writing animation (line-by-line instead of all at once)
                for line in assistant_reply.split('\n'):
                    st.markdown(line)
                    time.sleep(0.05)  # Add slight delay between lines

            except Exception as e:
                assistant_reply = f"⚠️ Error: {e}"
                st.markdown(assistant_reply)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})