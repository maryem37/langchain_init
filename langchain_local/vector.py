from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document  # <- corrected

import os
import pandas as pd

# Lire le CSV
df = pd.read_csv("realistic_restaurant_reviews.csv")
print("Columns in CSV:", df.columns)  # Check column names

# Embeddings Ollama
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Emplacement base de données
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# Créer le vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        doc = Document(
            page_content=row["title"] + " " + row["Review"],  # <- adjust column names
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        documents.append(doc)
        ids.append(str(i))

    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.persist()

# Créer un retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Test query
query = "Which pizza is the spiciest?"
results = retriever.get_relevant_documents(query)
for doc in results:
    print(doc.page_content, doc.metadata)
    print("------")
