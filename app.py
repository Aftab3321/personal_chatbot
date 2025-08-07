import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
import time

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

# === Main UI ===
st.markdown("<h1 style='text-align: left;'>Chats</h1>", unsafe_allow_html=True)
user_input = st.chat_input("Type your message...")

# === Display Chat History ===
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         with st.chat_message("user"):
#             st.markdown(msg["content"])
#     else:
#         with st.chat_message("assistant"):
#             st.markdown(msg["content"])

# Display previous messages
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
                response = llm.invoke(chat_history)
                assistant_reply = response.content

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
