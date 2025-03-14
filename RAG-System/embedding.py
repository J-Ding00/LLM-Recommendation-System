import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Define a function to create the client only when needed
def get_openai_client():
    """Lazy initialization: Creates and returns the OpenAI client once."""
    global client
    if "client" not in globals():
        client = OpenAI(api_key=api_key)  # Create the client only once
    return client

def get_embedding(text):
    """Generate an embedding for a given text."""
    client = get_openai_client()
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    # return response
    return response.data[0].embedding
