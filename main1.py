import streamlit as st
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import os
import fitz  # PyMuPDF
from groq import Groq
import pandas as pd
from duckduckgo_search import DDGS

# Initialize Groq client with your API key
groq_client = Groq(api_key="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")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit app setup
st.title("Retrieval-Augmented Generation (RAG) System with Research Paper and Web Search")
st.write(
    "This RAG system uses DistilBERT for embedding PDF documents and LLaMA from Groq for response generation based on query-relevant documents or web search results."
)

# Function to extract text from PDFs in a directory
def load_documents_from_directory(pdf_directory):
    pdf_texts = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            with fitz.open(file_path) as pdf_document:
                text = ""
                for page in pdf_document:
                    text += page.get_text()
            pdf_texts.append(text)
    return pdf_texts

# Load DistilBERT for embeddings
@st.cache_resource
def load_distilbert_model():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

# Embed the documents using DistilBERT
def embed_documents(docs, tokenizer, model):
    inputs = tokenizer(docs, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]  # Use CLS token representation
    return embeddings.numpy()

# Initialize FAISS index
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # The dimension of embeddings
    index.add(embeddings.astype('float32'))  # Ensure embeddings are in float32
    return index

# Retrieve relevant documents based on the query embedding
def retrieve(query_text, faiss_index, doc_texts, tokenizer, model, k=1):
    query_embedding = embed_documents([query_text], tokenizer, model)
    distances, indices = faiss_index.search(query_embedding.astype('float32'), k)
    return [doc_texts[i] for i in indices[0]]  # Return the retrieved documents

# Generate a response using LLaMA from Groq
def generate_response(retrieved_docs, query_text, groq_client, max_tokens=100):
    if not retrieved_docs or not any(retrieved_docs):
        return "No relevant documents found to generate a response."

    input_text = " ".join(retrieved_docs) + " " + query_text  # Concatenate retrieved documents with query

    # Prepare the messages for the Groq API
    messages = [
        {"role": "user", "content": input_text}
    ]

    # Call Groq API to generate response
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=1,
            max_tokens=max_tokens,
            top_p=1,
            stream=False,  # Set to False to receive the entire response at once
            stop=None,
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {str(e)}"

# Function to retrieve DuckDuckGo search results and convert them into embeddings
def retrieve_search_results(query_text, tokenizer, model):
    ddgs = DDGS()
    search_results = ddgs.text(
        keywords=query_text,
        region='wt-wt',  # Global region
        safesearch='off',  # Safe search off
        timelimit='7d',  # Results from the last 7 days
        max_results=3  # Limit to 3 results
    )

    # Extract the text and URLs from the search results
    result_texts = [result['title'] + " " + result['body'] for result in search_results]
    result_links = [result['href'] for result in search_results]

    # Embed the results using DistilBERT
    result_embeddings = embed_documents(result_texts, tokenizer, model)

    return result_texts, result_embeddings, result_links

# Main logic of the app
def rag_pipeline(query_text, docs_texts, groq_client):
    # Load models
    distilbert_tokenizer, distilbert_model = load_distilbert_model()

    # Retrieve and embed search results
    search_texts, search_embeddings, search_links = retrieve_search_results(query_text, distilbert_tokenizer, distilbert_model)

    # Embed PDF documents and build FAISS index
    doc_embeddings = embed_documents(docs_texts, distilbert_tokenizer, distilbert_model)
    combined_embeddings = torch.cat([torch.tensor(doc_embeddings), torch.tensor(search_embeddings)])

    faiss_index = create_faiss_index(combined_embeddings.numpy())

    # Combine document texts with search results
    combined_texts = docs_texts + search_texts

    # Retrieve relevant documents based on query
    retrieved_docs = retrieve(query_text, faiss_index, combined_texts, distilbert_tokenizer, distilbert_model)

    # Generate response using the retrieved documents as context
    response = generate_response(retrieved_docs, query_text, groq_client, max_tokens=100)

    return response, search_links  # Return the response and the search result links

# Streamlit user input
directory_path_input = r"E:\SecureGen\rag\document"  # Change this path as needed
query_input = st.text_input("Enter your query:")

if directory_path_input and query_input:
    with st.spinner("Loading and embedding documents..."):
        loaded_docs = load_documents_from_directory(directory_path_input)  # Load documents
        if loaded_docs:
            with st.spinner("Generating response..."):
                final_response, references = rag_pipeline(query_input, loaded_docs, groq_client)
                st.write("**Response:**")
                st.write(final_response)

                st.write("**References:**")
                for link in references:
                    st.write(link)
        else:
            st.write("No documents found in the provided directory.")
