import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

# Define a function to create the client only when needed
def get_pinecone_client():
    """Lazy initialization: Creates and returns the PineCone client once."""
    global client
    if "client" not in globals():
        client = Pinecone(api_key=api_key)  # Create the client only once
    return client

def do():
    client = get_pinecone_client()