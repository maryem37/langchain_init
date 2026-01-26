import streamlit as st
import requests
import os
# Import your helper functions from document_loader.py
from document_loader import process_document, store_embeddings


API_URL = "http://127.0.0.1:8000/query"

# Set Streamlit page config
st.set_page_config(page_title="AI-Powered Knowledge Assistant", page_icon="🤖")

# Title
st.title("📚 AI-Powered Knowledge Assistant")

# Sidebar for file upload
st.sidebar.header("📂 Upload Documents")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Save uploaded file
#creates a file path for storing the uploaded document in an uploaded files directory
# Save uploaded file
if uploaded_file:
    # Ensure folder exists
    os.makedirs("uploaded_files", exist_ok=True)

    # Save the file
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully!")

    # Process the uploaded file and store embeddings
    texts = process_document(file_path)
    if texts:
        store_embeddings(texts)
        st.sidebar.success("📄 Document processed and embeddings stored!")



# Chat-like UI
st.subheader("💬 Ask a Question")
user_query = st.text_input("Type your question:")

# Send query to API
if st.button("Ask AI"):
    if user_query:
        response = requests.post(API_URL, json={"query": user_query})
        answer = response.json().get("response", "No response available.")
        st.markdown(f"**🤖 AI Response:** {answer}")
    else:
        st.warning("Please enter a question.")





