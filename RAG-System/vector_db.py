from pinecone import PineconeException

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

