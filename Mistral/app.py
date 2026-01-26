from fastapi import FastAPI
from pydantic import BaseModel

# Correct Chroma import
from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# Initialize FastAPI app
app = FastAPI()

# Load Mistral AI via Ollama
llm = OllamaLLM(model="mistral")

# Load the embeddings model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Load ChromaDB
vectorstore = Chroma(
    collection_name="documents",
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Request model
class QueryRequest(BaseModel):
    query: str

# POST endpoint
@app.post("/query")
def search_and_generate_response(request: QueryRequest):
    response = qa_chain.invoke(request.query)
    return {"query": request.query, "response": response}

# Root endpoint
@app.get("/")
def home():
    return {"message": "Mistral AI-powered search API is running!"}
