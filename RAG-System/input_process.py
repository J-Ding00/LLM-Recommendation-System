import pypdf
import tiktoken
import pandas as pd
import os
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

# Define a function to create the client only when needed
def get_openai_client():
    """Lazy initialization: Creates and returns the OpenAI client once."""
    global openai_client
    if "openai_client" not in globals():
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=openai_api_key)  # Create the client only once
    return openai_client

# Define a function to create the client only when needed
def get_pinecone_index(index_name='llm-recommendation-system'):
    """Lazy initialization: Creates and returns the PineCone client once."""
    global pinecone_client
    if "pinecone_client" not in globals():
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_client = Pinecone(api_key=pinecone_api_key)  # Create the client only once
        if not pinecone_client.has_index(index_name):
            pinecone_client.create_index(
                name=index_name,
                vector_type="dense",
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                deletion_protection="disabled"
            )
    global pinecone_index
    if "pinecone_index" not in globals():
        pinecone_index = pinecone_client.Index(host=pinecone_client.describe_index(name=index_name)['host'])
    return pinecone_index

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file, page by page"""
    pdf_reader = pypdf.PdfReader(pdf_path)
    text_data = []

    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        if page_text:  # Ignore empty pages
            text_data.append(page_text)

    return " ".join(text_data)  # Join all pages into one long text

def split_text_with_overlap(text, max_tokens=300, overlap=50):
    """Splits text into chunks with a defined overlap to preserve boundary context"""
    tokenizer = tiktoken.get_encoding("o200k_base")  # GPT-4o/GPT-4o Mini tokenizer
    tokens = tokenizer.encode(text)  # Tokenize the full text

    chunks = []
    start = 0  # Sliding window start position

    while start < len(tokens):
        end = start + max_tokens  # Define chunk boundary
        chunk_tokens = tokens[start:end]  # Extract chunk tokens
        chunk_text = tokenizer.decode(chunk_tokens)  # Convert back to text
        chunks.append(chunk_text)

        # Move window forward, keeping an overlap
        start += max_tokens - overlap  # Shift by (chunk size - overlap)

    return chunks

# def get_embedding(text_chunks):
#     """Generate an embedding for a given text."""
#     client = get_openai_client()
    
#     response = client.embeddings.create(
#         input=text_chunks,
#         model="text-embedding-3-small"
#     )
#     # return response
#     print(response)
#     return response.data[0].embedding

def get_embedding(text_chunks):
    """Generate embeddings for a batch of text inputs using text-embedding-3-small model."""
    all_embeddings = []
    batch_size = len(text_chunks)
    # Process texts in  batches
    client = get_openai_client()
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]  # Get batch of texts
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch  # Pass the batch of texts
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

def batch_embed_upsert(text_chunks, file_name='', chunk_token_size=300, max_input_token=8191, pinecone_index='llm-recommendation-system', pinecone_index_namespace=''):
    """Stores text embeddings in Pinecone using batch processing."""
    batch_size = max_input_token // chunk_token_size
    index = get_pinecone_index(pinecone_index)

    for i in range(0, len(text_chunks), batch_size):
        batch_texts = text_chunks[i:i + batch_size]
        batch_embeddings = get_embedding(batch_texts)
        # print(batch_embeddings)

        # Prepare batch upsert data
        upsert_data = [
            {'id': file_name+'#'+str(i + j), 'values':batch_embeddings[j], 'metadata':{"text": batch_texts[j]}}
            for j in range(len(batch_texts))
        ]
        
        index.upsert(upsert_data, namespace=pinecone_index_namespace)

def process_pdf(pdf_path, chunk_size, overlap, max_chunk_len, pinecone_index='llm-recommendation-system', pinecone_index_namespace='', reserve_data=False):
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text_with_overlap(pdf_text, max_tokens=chunk_size, overlap=overlap)[:max_chunk_len]
    batch_embed_upsert(text_chunks=text_chunks, file_name=pdf_path, chunk_token_size=chunk_size, pinecone_index=pinecone_index, pinecone_index_namespace=pinecone_index_namespace)
    # if not reserve_data:


# # temp
# def save_to_csv(chunks, embeddings, filename="text_embeddings.csv"):
#     """Saves text chunks and embeddings into a CSV file"""
#     df = pd.DataFrame({
#         "text": chunks,
#         "embedding": [str(embed) for embed in embeddings],  # Convert list to string
#         "n_tokens": [len(tiktoken.get_encoding("o200k_base").encode(chunk)) for chunk in chunks]
#     })
#     df.to_csv(filename, index=False)
#     print(f"Saved {len(chunks)} chunks to {filename}")