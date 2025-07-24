import truststore
truststore.inject_into_ssl()

import chromadb
from sentence_transformers import SentenceTransformer
import time
import json

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_DB_PATH = "./chroma_db_nougat_chapter9"
CHROMA_COLLECTION_NAME = "chapter9_nougat_content_v1"
N_RESULTS_TO_RETRIEVE = 3
OUTPUT_RETRIEVAL_RESULTS_FILE = "./data/retrieval_test_results.json"

print(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Sentence transformer model loaded successfully.")
except Exception as e:
    print(f"Error loading sentence transformer model: {e}")
    exit()

print(f"\nConnecting to ChromaDB at path: {CHROMA_DB_PATH}")
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    print(f"Successfully connected to collection '{CHROMA_COLLECTION_NAME}'.")
    print(f"Collection contains {collection.count()} documents.")
    if collection.count() == 0:
        print("Warning: Collection is empty! Ensure Day 2 script ran successfully and populated the DB.")
except Exception as e:
    print(f"Error connecting to ChromaDB or getting collection: {e}")
    print("Ensure the DB path and collection name are correct and Day 2 script ran successfully.")
    exit()

def retrieve_relevant_chunks(query_text, emb_model, db_collection, n_results=N_RESULTS_TO_RETRIEVE):
    """
    Retrieves relevant text chunks from ChromaDB based on a query.

    Args:
        query_text (str): The user's query.
        emb_model (SentenceTransformer): The loaded sentence embedding model.
        db_collection (chromadb.api.models.Collection.Collection): The ChromaDB collection object.
        n_results (int): The number of top results to retrieve.

    Returns:
        dict: A dictionary containing the query results from ChromaDB,
              typically including 'ids', 'documents', 'metadatas', 'distances'.
              Returns None if an error occurs.
    """
    print(f"\nRetrieving chunks for query: '{query_text}'")
    try:
        query_start_time = time.time()
        query_embedding_array = emb_model.encode([query_text])
        query_embedding_list = query_embedding_array[0].tolist()
        print(f"  Query embedding generated in {time.time() - query_start_time:.4f} seconds.")

        query_db_start_time = time.time()
        results = db_collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        print(f"  ChromaDB query completed in {time.time() - query_db_start_time:.4f} seconds.")
        return results
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return None

if __name__ == "__main__":
    sample_queries = [
        "What is margin of error?",
        # "Explain the concept of a pivotal quantity.",
        # "How is the t-distribution used in confidence intervals?",
        # "Describe the properties of maximum likelihood estimators.",
        "How to perform hypothesis test?",
        "What is the relationship between critical-value test and p-value test?",
        "what is the difference between Neyman-Pearson decision rule and hypothesis testing",
        # "Central Limit Theorem applications in estimation"
    ]

    all_retrieval_outputs = []

    print("\n--- Testing Retrieval Function & Collecting Results ---")
    retrieval_start_time = time.time()

    for i, query in enumerate(sample_queries):
        print(f"\n--- Processing Test Query {i+1}/{len(sample_queries)}: \"{query}\" ---")
        query_process_start_time = time.time()
        
        retrieval_results = retrieve_relevant_chunks(query, embedding_model, collection, n_results=N_RESULTS_TO_RETRIEVE)
        
        query_output = {
            "query_id": f"query_{i+1}",
            "query_text": query,
            "retrieved_results_raw": retrieval_results,
            "retrieved_items": []
        }

        if retrieval_results and retrieval_results.get('documents') and retrieval_results.get('documents')[0]:
            ids_list = retrieval_results.get('ids', [[]])[0]
            documents_list = retrieval_results.get('documents', [[]])[0]
            metadatas_list = retrieval_results.get('metadatas', [[]])[0]
            distances_list = retrieval_results.get('distances', [[]])[0]

            print(f"  Retrieved {len(documents_list)} results:")
            for j in range(len(documents_list)):
                item = {
                    "rank": j + 1,
                    "id": ids_list[j],
                    "distance": float(distances_list[j]),
                    "metadata": metadatas_list[j],
                    "document_text": documents_list[j]
                }
                query_output["retrieved_items"].append(item)
                
                print(f"\n    Result {j+1}:")
                print(f"      ID: {item['id']}")
                print(f"      Distance: {item['distance']:.4f}")
                print(f"      Metadata: {json.dumps(item['metadata'], indent=2)}")
                print(f"      Document Preview (first 300 chars):")
                preview_text = item['document_text'][:300].replace('\n', ' ')
                print(f"        \"{preview_text}...\"")
        else:
            print("    No results returned or an error occurred during retrieval.")
            query_output["retrieved_items"] = None 

        all_retrieval_outputs.append(query_output)
        print(f"  Query processing time: {time.time() - query_process_start_time:.2f} seconds.")

    print(f"\nTotal time for all retrievals: {time.time() - retrieval_start_time:.2f} seconds.")

    print(f"\nSaving all retrieval results to '{OUTPUT_RETRIEVAL_RESULTS_FILE}'...")
    try:
        with open(OUTPUT_RETRIEVAL_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_retrieval_outputs, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved retrieval results to '{OUTPUT_RETRIEVAL_RESULTS_FILE}'.")
    except Exception as e:
        print(f"Error saving retrieval results to JSON: {e}")
