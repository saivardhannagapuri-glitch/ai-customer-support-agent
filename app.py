
import streamlit as st
import requests

st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="💬",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/ask"

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
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get(
                    "answer",
                    "Sorry, I couldn't find an answer."
                )
                sources = data.get("sources", [])

            except requests.RequestException:
                answer = (
                    "I couldn't connect to the support service. "
                    "Please make sure the API server is running."
                )
                sources = []

            st.write(answer)

            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.write(f"- {source}")

    # Save the assistant response in conversation history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })