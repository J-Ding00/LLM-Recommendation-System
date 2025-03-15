import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

# Define a function to create the client only when needed
def get_pinecone_index(index_name='llm-recommendation-system'):
    """Lazy initialization: Creates and returns the PineCone client once."""
    global pinecone_client
    if "pinecone_client" not in globals():
        api_key = os.getenv("PINECONE_API_KEY")
        pinecone_client = Pinecone(api_key=api_key)  # Create the client only once
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

def pinecone_query(query, top_k=3,namespace=''):
    pinecone_index = get_pinecone_index()
    results = pinecone_index.query(
        namespace=namespace,
        vector=query, 
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )
    return results

def clear_pinecone_by_namespace(namespace=''):
    # client = get_pinecone_client()
    pinecone_index = get_pinecone_index()
    pinecone_index.delete(delete_all=True, namespace=namespace)

def clear_pinecone_by_filename(namespace='', file_name=''):
    pinecone_index = get_pinecone_index()
    for ids in pinecone_index.list(prefix=file_name+'#', namespace=namespace):
        # print(ids) # ['doc1#chunk1', 'doc1#chunk2', 'doc1#chunk3']
        pinecone_index.delete(ids=ids, namespace=namespace)

