from pinecone import PineconeException

# Load environment variables once
# load_dotenv()

# Define a function to create the client only when needed
# def get_pinecone_index(index_name='llm-recommendation-system'):
#     """Lazy initialization: Creates and returns the PineCone client once."""
#     global pinecone_client
#     if "pinecone_client" not in globals():
#         api_key = os.getenv("PINECONE_API_KEY")
#         pinecone_client = Pinecone(api_key=api_key)  # Create the client only once
#         if not pinecone_client.has_index(index_name):
#             pinecone_client.create_index(
#                 name=index_name,
#                 vector_type="dense",
#                 dimension=1536,
#                 metric="cosine",
#                 spec=ServerlessSpec(
#                     cloud="aws",
#                     region="us-east-1"
#                 ),
#                 deletion_protection="disabled"
#             )
#     global pinecone_index
#     if "pinecone_index" not in globals():
#         pinecone_index = pinecone_client.Index(host=pinecone_client.describe_index(name=index_name)['host'])
#     return pinecone_index

def pinecone_query(index, namespace, query, top_k):
    results = index.query(
        namespace=namespace,
        vector=query, 
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )
    return results

def clear_pinecone_by_namespace(index, namespace):
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"Deleted Namespace '{namespace}'.")
    except PineconeException as e:
        if "Namespace not found" in str(e):
            print(f"Namespace '{namespace}' not found, skipping deletion.")
        else:
            raise

def clear_pinecone_by_filename(index, namespace, file_name):
    # pinecone_index = get_pinecone_index()
    try:
        num_record = 0
        for ids in index.list(prefix=file_name+'#', namespace=namespace):
        # print(ids) # ['doc1#chunk1', 'doc1#chunk2', 'doc1#chunk3']
            index.delete(ids=ids, namespace=namespace)
            num_record += len(ids)
        print(f'Deleted {num_record} records.')
    except PineconeException as e:
        if "Namespace not found" in str(e):
            print(f"Namespace '{namespace}' not found, skipping deletion.")
        else:
            raise

