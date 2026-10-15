
import streamlit as st
from src.langgraph_agent import graph

st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="💬",
    layout="centered",
)

st.title("💬 AI Customer Support Agent")
st.caption(
    "Ask questions about customer support, orders, refunds, and accounts."
)

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.subheader("Conversation")
    st.write(f"Messages: {len(st.session_state.messages)}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.markdown(f"- {source}")

# Chat input
question = st.chat_input("Ask a customer support question...")

if question:
    # Display and save the user's question
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate the assistant response
    with st.chat_message("assistant"):
        with st.spinner("Finding the best answer..."):
            try:
                result = graph.invoke({
                    "question": question,
                    "retrieved_documents": [],
                    "answer": "",
                    "sources": [],
                })

                answer = result.get("answer") or (
                    "Sorry, I couldn't find an answer."
                )
                sources = result.get("sources") or []

                # Normalize sources for display
                sources = [
                    str(source).strip()
                    for source in sources
                    if source and str(source).strip()
                ]

            except Exception as error:
                error_message = str(error)

                print(
                    f"Customer support agent error: {error_message}"
                )

                # Handle Gemini API quota errors
                if (
                    "RESOURCE_EXHAUSTED" in error_message
                    or "429" in error_message
                    or "quota" in error_message.lower()
                ):
                    answer = (
                        "The AI support service has temporarily "
                        "reached its usage limit. Please try again "
                        "later. Your question has not been processed."
                    )
                else:
                    answer = (
                        "The support agent couldn't process this "
                        "request. Please try again later. If the "
                        "issue continues, contact customer support."
                    )

                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.markdown(f"• {source}")

    # Save the complete assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })