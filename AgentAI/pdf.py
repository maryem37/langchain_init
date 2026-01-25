from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

def get_index(pdf_path, name):
    # 1️⃣ Load documents
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

    # 2️⃣ Create a node parser to split documents into chunks
    parser = SimpleNodeParser(
        chunk_size=256,
        chunk_overlap=50
    )

    # 3️⃣ Convert documents into nodes (pre-chunked)
    nodes = parser.get_nodes_from_documents(documents)

    # 4️⃣ Ollama embeddings
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text"
    )

    # 5️⃣ Create index
    index = VectorStoreIndex.from_documents(
        nodes,
        embed_model=embed_model,
        show_progress=True
    )

    # 6️⃣ Create LLM for query engine ✅ ADD THIS
    # In the get_index function, change the LLM line to:
    llm = Ollama(
    model="llama3.2:1b", 
    request_timeout=120.0,
    system_prompt="Provide direct, concise answers based on the context. No conversational extras."
    )   
    
    # 7️⃣ Return query engine with Ollama LLM ✅ FIX THIS
    return index.as_query_engine(llm=llm)


# Create the query engine
canada_engine = get_index("data/canada.pdf", "canada")