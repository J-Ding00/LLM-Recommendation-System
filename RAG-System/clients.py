import os
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import yaml

# Load environment variables from .env
load_dotenv()

# Setup OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

with open('./rag-system/config.yaml', "r") as file:
    config = yaml.safe_load(file)

# Define the index name and check if the index exists; create if not
if not pinecone_client.has_index(config['pinecone']['index_name']):
    pinecone_client.create_index(
        name=config['pinecone']['index_name'],
        vector_type="dense",
        dimension=config['pinecone']['dimension'],
        metric=config['pinecone']['metric'],
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled"
    )

# Connect to the Pinecone index
pinecone_index = pinecone_client.Index(host=pinecone_client.describe_index(name=config['pinecone']['index_name'])['host'])