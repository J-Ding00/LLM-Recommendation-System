import pypdf
import tiktoken
import embedding

def extract_text_from_pdf(pdf_path):
    try:
        """Extracts text from a PDF file, page by page"""
        pdf_reader = pypdf.PdfReader(pdf_path)
        text_data = []

        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            if page_text:  # Ignore empty pages
                text_data.append(page_text)

        return " ".join(text_data)  # Join all pages into one long text
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}.")

def split_text_with_overlap(tokenizer_name, text, max_tokens, overlap, max_chunk_len):
    """Splits text into chunks with a defined overlap to preserve boundary context"""
    tokenizer = tiktoken.get_encoding(tokenizer_name)  # GPT-4o tokenizer
    tokens = tokenizer.encode(text)  # Tokenize the full text

    chunks = []
    start = 0  # Sliding window start position
    num_chunk = 0

    while start < len(tokens) and num_chunk < max_chunk_len:
        end = start + max_tokens  # Define chunk boundary
        chunk_tokens = tokens[start:end]  # Extract chunk tokens
        chunk_text = tokenizer.decode(chunk_tokens)  # Convert back to text
        chunks.append(chunk_text)
        num_chunk += 1

        # Move window forward, keeping an overlap
        start += max_tokens - overlap  # Shift by (chunk size - overlap)

    return chunks

def batch_embed_upsert(text_chunks, file_name, chunk_token_size, embed_model, max_input_token, openai_client, pinecone_index, pinecone_index_namespace):
    """Stores text embeddings in Pinecone using batch processing."""
    batch_size = max_input_token // chunk_token_size

    for i in range(0, len(text_chunks), batch_size):
        batch_texts = text_chunks[i:i + batch_size]
        batch_embeddings = embedding.get_batch_embedding(openai_client, batch_texts, embed_model)
        # print(batch_embeddings)

        # Prepare batch upsert data
        upsert_data = [
            {'id': file_name+'#'+str(i + j), 'values':batch_embeddings[j], 'metadata':{"text": batch_texts[j]}}
            for j in range(len(batch_texts))
        ]
        
        pinecone_index.upsert(upsert_data, namespace=pinecone_index_namespace)
    print(f"File '{file_name}' inserted")

def process_pdf(pdf_path, chunk_size, overlap, max_chunk_len, embedding_model, max_embedding_input_token, tokenizer_name, openai_client, pinecone_index, pinecone_index_namespace):
    pdf_text = extract_text_from_pdf(pdf_path)
    if pdf_text:
        text_chunks = split_text_with_overlap(tokenizer_name, pdf_text, chunk_size, overlap, max_chunk_len=max_chunk_len)
        batch_embed_upsert(text_chunks, pdf_path, chunk_size, embedding_model, max_embedding_input_token, openai_client, pinecone_index, pinecone_index_namespace)
        return True
    else:
        return False