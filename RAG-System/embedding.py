# improve: generator
def get_query_embedding(client, query, model_name):
    """Generate an embedding for a given text."""  
    response = client.embeddings.create(
        input=query,
        model=model_name
    )
    return response.data[0].embedding

def get_batch_embedding(client, text_chunks, model_name):
    """Generate embeddings for a batch of text inputs using text-embedding-3-small model."""
    all_embeddings = []
    batch_size = len(text_chunks)
    # Process texts in  batches
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]  # Get batch of texts
        
        response = client.embeddings.create(
            model=model_name,
            input=batch  # Pass the batch of texts
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

def answer_question_with_rag(client, query, context, model_name):
    """Uses GPT-4o to answer a question based on retrieved and reranked text"""
    context = "\n".join(context)  # Combine top paragraphs

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an AI assistant that provides answers based on retrieved text. If you find the retrieved text relavant, clearly list the source. Otherwise, say that you can't answer given the context"},
            {"role": "user", "content": f"Using the following information, answer the question:\n\n{context}\n\nQ: {query}"}
        ]
    )
    return response.choices[0].message.content