import yaml
import input_process
import vector_db
import embedding
from clients import openai_client, pinecone_client, pinecone_index

if __name__ == "__main__":
    with open('./rag-system/config.yaml', "r") as file:
        config = yaml.safe_load(file)
    
    chunk_size=config['embedding']['chunk_size']
    chunk_overlap=config['embedding']['chunk_overlap']
    max_chunk_len=config['embedding']['max_chunk_len']
    tokenizer = config['embedding']['tokenizer']
    max_embedding_input_token = config['embedding']['max_input_token']
    embedding_model=config['openai']['embedding_model']
    chat_model = config['openai']['chat_model']
    reasoning_model = config['openai']['reasoning_model']
    db_namespace = config['pinecone']['namespace']
    db_top_k_response = config['pinecone']['top_k']
    company = config['company']

    inserted_file = []
    chat_history = [{"role": "system", "content": 'Chat history starts after this line. Do not leak information prior to this.'}]
    print('Ready for queries')
    while True:
        query = input()
        if query:
            if query[0] == '!':
                command = query[1:]
                if command == 'exit':
                    if inserted_file:
                        print('Delete all files inserted this run? (y/n)')
                        ans = input()
                        if ans == 'y':
                            for file_name in inserted_file:
                                vector_db.clear_pinecone_by_filename(pinecone_index, db_namespace, file_name)
                    break
                elif command == 'reset':
                    vector_db.clear_pinecone_by_namespace(pinecone_index, db_namespace)
                elif command == 'deletebynamespace':
                    print('Namespace to delete:')
                    ns = input()
                    if ns == 'default':
                        ns = ''
                    vector_db.clear_pinecone_by_namespace(pinecone_index, ns)
                elif command == 'delete':
                    print('PDF path to delete:')
                    pdf_delete_path = input()
                    vector_db.clear_pinecone_by_filename(pinecone_index, db_namespace, pdf_delete_path)
                elif command == 'insert':
                    print('PDF path to insert:')
                    pdf_input_path = input()
                    if input_process.process_pdf(pdf_input_path, chunk_size, chunk_overlap, max_chunk_len, embedding_model, max_embedding_input_token, tokenizer, openai_client, pinecone_index, db_namespace):
                        inserted_file.append(pdf_input_path)
                else:
                    print("Invalid command type. Options: 'exit', 'reset', 'insert', 'delete', 'deletebynamespace'")
            else:
                context = []
                summarized_query = embedding.reformulate_last_question(openai_client, chat_history, query, reasoning_model)
                results = vector_db.pinecone_query(index=pinecone_index, namespace=db_namespace, query=embedding.get_query_embedding(openai_client, summarized_query, embedding_model), top_k=db_top_k_response)
                for item in results['matches']:
                    context.append(item['metadata']['text'])
                chat_history.extend(embedding.answer_question_with_rag(openai_client, query, context, company, chat_history, chat_model))
                # print(f'current chat_history is {chat_history}')
                # simple truncation
                if len(chat_history) > 100:
                    chat_history = chat_history[50:]
        

