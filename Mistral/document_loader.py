import os
import fitz  # PyMuPDF for PDFs
import docx
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_chroma import Chroma
from langchain.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA


# Load the embeddings model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Load the LLM (Mistral) using Ollama
llm = OllamaLLM(model="mistral")


def search_and_summarize(query, db_path="chroma_db"):
    """Retrieve relevant documents and summarize them using Mistral AI"""
    
    # Load ChromaDB
    vectorstore = Chroma(
        collection_name="documents",
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    
    # Create a LangChain Retrieval-QA pipeline
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Get AI-generated answer
    response = qa_chain.invoke(query)

    print("\n💡 AI-Powered Answer:")
    print(response.get("result"))


def generate_ai_response(context, query):
    """Send user query along with retrieved documents to Mistral AI for RAG"""
    prompt = f"""
    You are an AI assistant with access to the following information:
    
    {context}
    
    Based on this, answer the following question:
    {query}
    """
    #A payload is just the data you send to a web API in a request.
#Think of it as the message or instructions you give to the server.
    payload = {
                "model": "mistral",# which AI model to use
                "prompt": prompt,  # the text you want the AI to respond to
                "stream": False  # do you want the response streamed in real-time? False = wait for full answer
                
                }
    response = requests.post(OLLAMA_API_URL, json=payload)
    
    return response.json().get("response", "No response generated.")


#this search document is a function that takes a query , a search term or phrase as input and searchs
#for the most relevent document stored in embeddings in chroma db.
#db is the default directory where chroma db stroes the vector embeddings.
def search_and_generate_response(query, db_path="chroma_db"):
    """Retrieve relevant documents and use Mistral AI for contextual response"""

    #persist_directory = loads the existing vector database from chroma DB
    #embedding fuction = embedding model uses the same embedding model , which is the huggingface embedding to ensure consistency
    #betwwen stored model and the query
    vectorstore = Chroma(
        collection_name="documents",
        persist_directory=db_path,
        embedding_function=embedding_model
    )

    #this results equal to vector dot similarity search query and equal 3 
    #it converts the query into embeddings
    #find the top5 similar embeddings stored in the chrome DB.
    #return a list of the best matching
    results = vectorstore.similarity_search(query, k=3)  # Retrieve top 3 matches
    
    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Generate AI response using RAG
    ai_response = generate_ai_response(context, query)
    
    print("\n💡 AI-Powered Answer:")
    print(ai_response)


def process_document(file_path):
    """Extract text, split it, and convert to embeddings"""
    text = extract_text(file_path)
    
    if not text:
        return None

    # Split text into smaller chunks for better search performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_text(text)

    return texts


def store_embeddings(texts, db_path="chroma_db"):
    """Store text embeddings in ChromaDB"""
    vectorstore = Chroma(
        collection_name="documents",
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    vectorstore.add_texts(texts)
    
    print("✅ Embeddings stored successfully!")


def search_documents(query, db_path="chroma_db"):
    """Search stored embeddings in ChromaDB"""
    vectorstore = Chroma(
        collection_name="documents",
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    results = vectorstore.similarity_search(query, k=3)  # Retrieve top 3 matches
    
    for idx, result in enumerate(results):
        print(f"\n🔹 Result {idx + 1}:")
        print(result.page_content)

    return results


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text


def extract_text_from_word(doc_path):
    """Extract text from a Word (.docx) file"""
    text = ""
    try:
        doc = docx.Document(doc_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading Word file {doc_path}: {e}")
    return text


def extract_text_from_txt(txt_path):
    """Extract text from a TXT file"""
    text = ""
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading TXT file {txt_path}: {e}")
    return text


def extract_text(file_path):
    """Detect file type and extract text accordingly"""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_word(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return ""


# Example usage
if __name__ == "__main__":
     #sample_file = "sample.pdf"
     #texts = process_document(sample_file)
     #if texts:
         #store_embeddings(texts)

     user_query = input("Enter your searh query: ")
    # results = search_documents(user_query)
    # search_and_generate_response(user_query)
     search_and_summarize(user_query)


