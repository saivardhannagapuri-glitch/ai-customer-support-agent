# AI Customer Support Agent

An AI-powered customer support application that answers questions using a company FAQ knowledge base. It combines Retrieval-Augmented Generation (RAG), LangGraph, Google Gemini, FastAPI, and Streamlit.

## Features

* Retrieves relevant information from the FAQ knowledge base.
* Uses embeddings and FAISS vector search to find relevant FAQ content.
* Uses Google Gemini to generate context-grounded responses.
* Uses LangGraph to manage the answer and fallback flow.
* Provides source references for answers.
* Includes a FastAPI backend and Streamlit user interface.
* Maintains chat history during the current Streamlit session.
* Includes basic evaluation tests.

## Technology Stack

* Python
* Retrieval-Augmented Generation (RAG)
* LangChain
* LangGraph
* Google Gemini
* Sentence Transformers
* FAISS
* FastAPI
* Streamlit

## Project Structure

```text
ai_customer_support_agent/
├── app.py
├── data/
│   └── faq.txt
├── evaluation/
│   └── test_agent.py
├── src/
│   ├── api.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── faiss_store.py
│   ├── generator.py
│   ├── langchain_rag.py
│   ├── langgraph_agent.py
│   ├── retriever.py
│   └── test_rag.py
├── requirements.txt
└── .gitignore
```

## Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/saivardhannagapuri-glitch/ai-customer-support-agent.git
   cd ai-customer-support-agent
   ```

2. Create and activate a Python virtual environment:

   ```bash
   py -3.12 -m venv venv
   ```

   Windows:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your own Gemini API key:

   ```text
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

   Never upload your real API key or `.env` file to GitHub.

## Run the Application

Start the Streamlit interface:

```bash
python -m streamlit run app.py
```

Start the FastAPI backend in a separate terminal:

```bash
python -m uvicorn src.api:app --reload
```

Open the Streamlit URL shown in the terminal. The FastAPI documentation is available at `http://127.0.0.1:8000/docs`.

## Evaluation

The initial evaluation suite passed **4 out of 4 selected test cases** covering password reset, refund duration, order status, and an unrelated question.

This is a basic test result, not a guarantee of accuracy for every possible question.

Run the evaluation with:

```bash
python -m evaluation.test_agent
```

## Future Improvements

* Deploy the application online.
* Expand the evaluation dataset.
* Improve retrieval and response quality.
* Add persistent conversation storage.

## Author

Saivardhan Nagapuri
