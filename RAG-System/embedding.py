import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

# Define a function to create the client only when needed
def get_openai_client():
    """Lazy initialization: Creates and returns the OpenAI client once."""
    global openai_client
    if "openai_client" not in globals():
        api_key = os.getenv("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=api_key)  # Create the client only once
    return openai_client

# improve: generator
def get_query_embedding(query):
    """Generate an embedding for a given text."""
    openai_client = get_openai_client()
    
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    # return response
    return response.data[0].embedding

def answer_question_with_rag(query, context):
    """Uses GPT-4o to answer a question based on retrieved and reranked text"""
    openai_client = get_openai_client()
    context = "\n".join(context)  # Combine top paragraphs

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that provides answers based on retrieved text. If you find the retrieved text relavant, clearly list the source. Otherwise, say that you can't answer given the context"},
            {"role": "user", "content": f"Using the following information, answer the question:\n\n{context}\n\nQ: {query}"}
        ]
    )
    return response.choices[0].message.content