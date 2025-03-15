import pypdf
import tiktoken
import pandas as pd

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

# temp
def save_to_csv(chunks, embeddings, filename="text_embeddings.csv"):
    """Saves text chunks and embeddings into a CSV file"""
    df = pd.DataFrame({
        "text": chunks,
        "embedding": [str(embed) for embed in embeddings],  # Convert list to string
        "n_tokens": [len(tiktoken.get_encoding("o200k_base").encode(chunk)) for chunk in chunks]
    })
    df.to_csv(filename, index=False)
    print(f"Saved {len(chunks)} chunks to {filename}")