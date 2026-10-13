from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


def generate_answer(query, retrieved_chunks):
    """
    Generate an answer using the retrieved context.
    """

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an AI customer support agent.

Answer the user's question using only the information provided in the context.

Context:

{context}

User question:

{query}

If the answer is not available in the context, say:

"I don't have enough information to answer that question."

Answer clearly and professionally.
"""

    response = model.invoke(prompt)

    return response.content