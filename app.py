import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, UnstructuredExcelLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import requests
import json

# Initialize HuggingFaceEmbeddings for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")

# Directory for Chroma DB persistence
persist_directory = "Folder\\Chroma_DB"

# Load and process both PDF and Excel files from the directory
folder_path = 'YOUR_FILE_PATH'
pdf_loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
excel_loader = DirectoryLoader(folder_path, glob="*.xlsx", loader_cls=UnstructuredExcelLoader)

# Load documents and filter out empty documents
pdf_documents = pdf_loader.load()
excel_documents = excel_loader.load()
documents = pdf_documents + excel_documents
documents = [doc for doc in documents if doc.page_content.strip()]  # Remove empty documents

# Ensure documents are not empty
if not documents:
    raise ValueError("No valid PDF or Excel documents found in the directory.")

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Ensure texts are not empty after splitting
if not texts:
    raise ValueError("Text splitting failed. No content available for processing.")

# Create Chroma DB using `Chroma.from_documents` with HuggingFaceEmbeddings
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding_model,
    persist_directory=persist_directory
)

# Reload the persisted database using `Chroma`
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

# Create a retriever
retriever = vectordb.as_retriever()

# LM Studio Code Functions
def verify_server():
    try:
        response = requests.get("http://localhost:1234/v1/models")
        response.raise_for_status()
        return True
    except:
        return False

def send_request(message, context):
    url = "http://localhost:1234/v1/chat/completions"
    
    # Combine context and user query
    full_message = f"Context: {context}\n\nUser Query: {message}"
    
    payload = {
        "model": "phi-3.1-mini-128k-instruct",
        "messages": [
            {"role": "system", "content": "Answer queries by utilizing the provided context. Respond accurately and concisely based on the retrieved information."},
            {"role": "user", "content": full_message}
        ],
        "temperature": 0.7,
        "max_new_tokens": 1024,
        "stream": False
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def format_response(response):
    if "error" in response:
        return response["error"]
    try:
        return response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return "Error: Unexpected response format"

# Ensure server is running
if not verify_server():
    print("Error: Cannot connect to LM Studio server")
else:
    print("LM Studio server connected. Starting query processing...")

    # User query loop
    while True:
        query = input("Enter your query (type 'exit' to stop): ")
        if query.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        
        try:
            # Retrieve relevant context from Chroma DB
            search_results = retriever.invoke(query)
            context = "\n".join([result.page_content for result in search_results])
            
            if not context.strip():
                print("No relevant context found in the database.")
                continue
            
            # Send the user query and context to the LM Studio server
            response = send_request(query, context)
            
            # Format and display the response
            print("Response:", format_response(response))
        
        except Exception as e:
            print(f"An error occurred: {e}")
