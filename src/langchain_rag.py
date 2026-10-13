from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from chunking import chunks


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Create Gemini model
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

print("Gemini model loaded successfully!")


# --------------------------------------------------
# 3. Create LangChain Documents
# --------------------------------------------------

documents = [
    Document(page_content=chunk)
    for chunk in chunks
]

print("Number of LangChain documents:", len(documents))


# --------------------------------------------------
# 4. Create embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# --------------------------------------------------
# 5. Create LangChain FAISS vector store
# --------------------------------------------------

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)

print("LangChain FAISS vector store created successfully!")


# --------------------------------------------------
# 6. Create retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

print("LangChain retriever created successfully!")


# --------------------------------------------------
# 7. Create Prompt Template
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are an AI customer support agent.

Answer the user's question using ONLY the information
provided in the context below.

Context:
{context}

User question:
{question}

If the answer is not available in the context, say:

"I don't have enough information to answer that question."

Answer clearly and professionally.
"""
)

print("Prompt template created successfully!")


# --------------------------------------------------
# 8. Format retrieved documents
# --------------------------------------------------

def format_docs(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


# --------------------------------------------------
# 9. Create complete LangChain RAG chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

print("LangChain RAG chain created successfully!")


# --------------------------------------------------
# 10. Ask a question
# --------------------------------------------------

query = "What is the capital of India?"


# --------------------------------------------------
# 11. Run complete RAG chain
# --------------------------------------------------

answer = rag_chain.invoke(query)


# --------------------------------------------------
# 12. Print final answer
# --------------------------------------------------

print("\nFinal answer:")
print(answer)