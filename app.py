
import streamlit as st
from src.langgraph_agent import graph

st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="💬",
    layout="centered"
)

st.title("💬 AI Customer Support Agent")
st.caption(
    "Ask questions about customer support, orders, refunds, and accounts."
)

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
with st.sidebar:
    st.subheader("Conversation")
    st.write(f"Messages: {len(st.session_state.messages)}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# Accept a new question
question = st.chat_input("Type your question here...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding the best answer..."):
            try:
                result = graph.invoke({
                    "question": question,
                    "route": "",
                    "retrieved_documents": [],
                    "answer": "",
                    "sources": []
                })

                answer = result.get(
                    "answer",
                    "Sorry, I couldn't find an answer."
                )
                sources = result.get("sources", [])

            except Exception:
                st.error(
                    "The support agent encountered an error. "
                    "Please check the app logs."
                )
                answer = (
                    "Sorry, I couldn't process your question right now."
                )
                sources = []

            st.write(answer)

            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.write(f"- {source}")

    # Save the assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })