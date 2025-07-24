import truststore
truststore.inject_into_ssl()

import json
import numpy as np # For array operations if needed, and embeddings are numpy arrays
from sentence_transformers import SentenceTransformer
import chromadb
import os
import time
import pandas as pd

def load_chunks_from_json(filepath):
    """Loads chunk data (list of dictionaries) from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        print(f"Successfully loaded {len(loaded_data)} chunks from '{filepath}'")
        return loaded_data
    except FileNotFoundError:
        print(f"Error: Chunk file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. File might be corrupted.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading chunks: {e}")
        return None

if __name__ == "__main__":
    start_script_time = time.time()

    CHUNK_FILE_PATH = "./data/chapter_9_nougat_chunks_with_metadata.json"
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
    CHROMA_DB_PATH = "./chroma_db_nougat_chapter9"
    CHROMA_COLLECTION_NAME = "chapter9_nougat_content_v1"

    print(f"\nStep 1: Loading chunks from '{CHUNK_FILE_PATH}'...")
    load_start_time = time.time()
    loaded_chunk_data = load_chunks_from_json(CHUNK_FILE_PATH)

    if not loaded_chunk_data:
        print("Exiting script as chunk data could not be loaded.")
        exit()

    text_chunks = [item['text'] for item in loaded_chunk_data]
    metadatas = [item['metadata'] for item in loaded_chunk_data]
    chunk_ids = [f"chunk_{i}" for i in range(len(text_chunks))]
    
    print(f"Loaded {len(text_chunks)} text chunks, {len(metadatas)} metadata entries, and {len(chunk_ids)} IDs.")
    print(f"Time for loading chunks: {time.time() - load_start_time:.2f} seconds.")

    print(f"\nStep 2: Generating embeddings using '{EMBEDDING_MODEL_NAME}'...")
    embedding_start_time = time.time()
    print(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Sentence transformer model loaded successfully.")
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}")
        print("Ensure you have an internet connection if downloading for the first time.")
        exit()
        
    print(f"\nGenerating embeddings for all {len(text_chunks)} text chunks...")
    print(f"This might take a few minutes...")
    try:
        # The encode method takes a list of strings and returns a NumPy array of embeddings
        chunk_embeddings_np = embedding_model.encode(text_chunks, show_progress_bar=True)
        
        print(f"Embeddings generated successfully. Shape of NumPy array: {chunk_embeddings_np.shape}")
        if chunk_embeddings_np.size > 0: # Check if not empty
             print(f"Dimension of first embedding vector: {chunk_embeddings_np[0].shape[0]}") # Should be e.g., 384
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        exit()
    print(f"Time for generating embeddings: {time.time() - embedding_start_time:.2f} seconds.")

    print(f"\nStep 3: Setting up ChromaDB at '{CHROMA_DB_PATH}' and ingesting data...")
    db_setup_start_time = time.time()
    print(f"Initializing ChromaDB persistent client at path: {CHROMA_DB_PATH}")
    try:
        # Create the directory for the DB if it doesn't exist
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        print(f"ChromaDB PersistentClient initialized. Data will be stored in: {CHROMA_DB_PATH}")
    except Exception as e:
        print(f"Error initializing ChromaDB client: {e}")
        exit()

    print(f"\nGetting or creating ChromaDB collection: '{CHROMA_COLLECTION_NAME}'")
    try:
        # To start fresh (e.g., if you changed chunking or model), you can delete first:
        # try:
        #     client.delete_collection(name=CHROMA_COLLECTION_NAME)
        #     print(f"Deleted existing collection '{CHROMA_COLLECTION_NAME}' for a fresh start.")
        # except Exception as del_exc:
        #     print(f"Note: Could not delete collection (it might not exist): {del_exc}")

        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{CHROMA_COLLECTION_NAME}' ready. Initial count: {collection.count()}")
    except Exception as e:
        print(f"Error getting or creating ChromaDB collection: {e}")
        exit()

    print(f"\nAdding {len(chunk_ids)} documents to collection '{CHROMA_COLLECTION_NAME}'...")
    try:
        collection.add(
            embeddings=chunk_embeddings_np.tolist(), # Convert numpy array to list of lists
            documents=text_chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )
        print("Data ingestion complete.")
        print(f"New collection count: {collection.count()}") # Verify count increased
    except Exception as e:
        print(f"Error adding data to ChromaDB collection: {e}")
        exit()
    print(f"Time for ChromaDB setup and ingestion: {time.time() - db_setup_start_time:.2f} seconds.")
    
    total_script_time = time.time() - start_script_time
    print(f"Total execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes).")
    print(f"Your vectorized chapter content is now stored in ChromaDB at '{CHROMA_DB_PATH}'.")
    print(f"Collection name: '{CHROMA_COLLECTION_NAME}'")
