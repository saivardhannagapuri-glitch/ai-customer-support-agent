from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.langgraph_agent import graph

app = FastAPI(
    title="AI Customer Support Agent API",
    description="An FAQ-based customer support agent powered by RAG and LangGraph.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Customer support question",
    )


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/")
def home():
    return {
        "message": "AI Customer Support Agent API is running!"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        result = graph.invoke({
            "question": question,
            "retrieved_documents": [],
            "answer": "",
            "sources": [],
        })

        return QuestionResponse(
            question=question,
            answer=result["answer"],
            sources=result.get("sources", []),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The agent encountered an internal error.",
        )
