import argparse
import input_process
import vector_db
import embedding

openai_client, pinecone_client, pinecone_index = None, None, None
if __name__ == "__main__":
    chunk_size=300
    overlap=50
    max_chunk_len=300
    parser = argparse.ArgumentParser()

    # Adding arguments
    parser.add_argument("-path", type=str)
    parser.add_argument("-reset", action="store_true")
    parser.add_argument("-reserve", action="store_true")
    parser.add_argument("-chunk_size", type=int)
    parser.add_argument("-overlap", type=int)
    parser.add_argument("-max_chunk_len", type=int)

    # Parse arguments
    args = parser.parse_args()
    if args.reset:
        vector_db.clear_pinecone_by_namespace()
    if args.chunk_size:
        chunk_size = args.chunk_size
    if args.overlap:
        overlap = args.overlap
    if args.max_chunk_len:
        args.max_chunk_len = args.max_chunk_len
    if args.path:
        input_process.process_pdf(pdf_path=args.path, chunk_size=chunk_size, overlap=overlap, max_chunk_len=max_chunk_len)

    print('Ready for queries')
    while True:
        query = input()
        if query == '!exit':
            break
        else:
            context = []
            results = vector_db.pinecone_query(query=embedding.get_query_embedding(query), top_k=3)
            for item in results['matches']:
                context.append(item['metadata']['text'])
            print(embedding.answer_question_with_rag(query, context))
            # print(results)
    
    if args.path and not args.reserve:
        vector_db.clear_pinecone_by_filename(args.path)
        

