import input_process
# import embedding

# def process_pdf_and_generate_embeddings(pdf_path):
#     """Extracts text from a PDF, splits it into chunks, and generates embeddings."""
#     pdf_text = input_process.extract_text_from_pdf(pdf_path)[:50]
#     text_chunks = input_process.split_text_with_overlap(pdf_text, 10, 5)

#     print(f"Generated {len(text_chunks)} chunks with boundary buffers")

#     embeddings = [embedding.get_embedding(chunk) for chunk in text_chunks]
#     print(f"Generated {len(embeddings)} embeddings")

#     return embeddings
openai_client, pinecone_client, pinecone_index = None, None, None
if __name__ == "__main__":
    pdf_path = "test/sample.pdf"
    
    input_process.process_pdf(pdf_path=pdf_path, chunk_size=30, overlap=5, max_chunk_len=3)

# import embedding
# import input_process
# import pandas as pd

# # response = embedding.get_embedding("Your text string goes here")
# # print(response)

# # Example usage
# pdf_text = input_process.extract_text_from_pdf("test/sample.pdf")
# # print(pdf_text[:200])  # Print first 500 characters to verify

# # Example Usage
# text_chunks = input_process.split_text_with_overlap(pdf_text)
# print(f"Generated {len(text_chunks)} chunks with boundary buffers")
# print(text_chunks)

# embeddings = [embedding.get_embedding(chunk) for chunk in text_chunks]
# print(f"Generated {len(embeddings)} embeddings")

# # Save the data
# input_process.save_to_csv(text_chunks, embeddings, 'test/test_csv')

# import numpy as np

# def cosine_similarity(vec1, vec2):
#     """Computes cosine similarity between two vectors"""
#     vec1, vec2 = np.array(vec1), np.array(vec2)
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def get_top_k_chunks(user_query, csv_file="test/test_csv.csv", k=3):
#     """Finds the top K most relevant text chunks based on cosine similarity"""
#     # Load stored embeddings
#     df = pd.read_csv(csv_file)
#     df["embedding"] = df["embedding"].apply(eval)  # Convert string back to list
    
#     # Convert user query to embedding
#     user_embedding = embedding.get_embedding(user_query)

#     # Compute similarity scores
#     df["similarity"] = df["embedding"].apply(lambda emb: cosine_similarity(emb, user_embedding))

#     # Get top K most similar chunks
#     top_chunks = df.nlargest(k, "similarity")["text"].tolist()
    
#     return top_chunks

# # Example usage
# query = "What are some upcoming music festivals?"
# top_chunks = get_top_k_chunks(query)
# print("Most relevant paragraphs:", top_chunks)

