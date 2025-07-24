import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import time
import json
import re
import os
import config

# --- Core Components Initialization ---
# Global variables to hold the initialized components
embedding_model = None
collection = None

def initialize_components():
    """
    Initializes and loads the embedding model and ChromaDB collection.
    This function should be called once when the application starts.
    """
    print("Loading sentence transformer model...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}")
        embedding_model = None

    print("Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"Connected to ChromaDB collection '{CHROMA_COLLECTION_NAME}' with {collection.count()} items.")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        collection = None
    
    return embedding_model, collection


# --- Helper & RAG Functions ---

def retrieve_relevant_chunks(query_text, embedding_model, collection, n_results=3):
    """
    Retrieves relevant text chunks from the globally defined ChromaDB collection.
    """
    if embedding_model is None or collection is None:
        print("Error: Components not initialized. Please call initialize_components() first.")
        return None
        
    try:
        query_embedding = embedding_model.encode([query_text])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] 
        )

        if results and results.get('distances') and results.get('distances')[0]:
            best_distance = results['distances'][0][0]
            print(f"  Relevance check: Best distance is {best_distance:.4f}")
            if best_distance > RELEVANCE_THRESHOLD:
                print("  Relevance check FAILED. Best match is not relevant enough.")
                return None 

        return results
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return None

def format_context_for_llm(retrieval_results_from_chroma, max_chunks_to_use=N_RESULTS_TO_RETRIEVE_FOR_RAG):
    """
    Formats retrieved chunks from ChromaDB into a single context string for the LLM.
    """
    context_parts = []
    if not (retrieval_results_from_chroma and 
            retrieval_results_from_chroma.get('documents') and 
            retrieval_results_from_chroma.get('documents')[0]):
        return "No relevant context found to provide to the LLM."

    documents_list = retrieval_results_from_chroma['documents'][0]
    metadatas_list = retrieval_results_from_chroma['metadatas'][0]
    
    num_to_use = min(len(documents_list), max_chunks_to_use)
    if num_to_use == 0:
        return "No relevant context found to provide to the LLM."

    context_parts.append("CONTEXT FROM TEXTBOOK ('Introduction to Probability for Data Science'):\n")
    for i in range(num_to_use):
        doc_text = documents_list[i]
        metadata = metadatas_list[i]
        source_info = "Unknown Section"
        if metadata:
            l3 = metadata.get('L3_Header')
            l2 = metadata.get('L2_Header')
            l1 = metadata.get('L1_Header')
            if l3: source_info = l3.replace("####","").strip() # Clean header markers
            elif l2: source_info = l2.replace("###","").strip()
            elif l1: source_info = l1.replace("##","").strip()
        
        context_parts.append(f"\n--- Retrieved Document {i+1} (Source: {source_info}) ---\n{doc_text}\n--- End of Document {i+1} ---")
    
    context_parts.append("\nEND OF PROVIDED CONTEXT.")
    return "\n".join(context_parts)
    
def parse_multiple_qa(llm_response_text, expected_count):
    """
    Parses the LLM response text to extract multiple Q&A pairs (Q1/A1, Q2/A2...).
    """
    qa_list = []
    pattern = re.compile(r"Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=\s*Q\d+:|\Z)", re.DOTALL | re.IGNORECASE)
    
    matches = pattern.finditer(llm_response_text)
    
    for match in matches:
        q_num = match.group(1)
        question = match.group(2).strip()
        answer = match.group(3).strip()
        qa_list.append({
            "question_num": int(q_num),
            "question": question,
            "answer": answer
        })
    
    if not qa_list:
        print("  Warning: Could not parse any Q&A pairs from LLM response using regex.")
        print(f"  Raw LLM response:\n{llm_response_text}")
    elif len(qa_list) < expected_count:
        print(f"  Warning: Expected {expected_count} Q&As but only parsed {len(qa_list)}.")
    else:
        print("  Successfully parsed Q&A set.")
        
    return qa_list

def generate_qa_via_rag(user_query, embedding_model, collection):
    """
    Generates a Question and Answer pair using the RAG pipeline.
    Uses globally defined components: retrieve_relevant_chunks, LLM_PROMPT_TEMPLATE, OLLAMA_MODEL_NAME.
    """
    print(f"\n--- RAG Pipeline for Query: '{user_query}' ---")

    print("Step 1: Retrieving context...")
    N_RESULTS_FOR_TOPIC = 5
    retrieval_time_start = time.time()
    retrieved_context_raw = retrieve_relevant_chunks(
        user_query, 
        embedding_model, 
        collection, 
        n_results=N_RESULTS_TO_RETRIEVE_FOR_RAG
    )
    print(f"  Context retrieval took {time.time() - retrieval_time_start:.2f}s")

    if not retrieved_context_raw:
        print("  Retrieval failed. Cannot generate Q&A.")
        return None, None, "Retrieval failed."

    print("Step 2: Formatting context...")
    formatted_context_for_llm = format_context_for_llm(retrieved_context_raw, max_chunks_to_use=N_RESULTS_FOR_TOPIC)
    
    if "No relevant context found" in formatted_context_for_llm:
        print("  No relevant context was found by retrieval. Cannot effectively prompt LLM.")
        return None, None, formatted_context_for_llm

    print("Step 3: Constructing LLM prompt...")
    full_prompt_for_llm = LLM_PROMPT_TEMPLATE.format(formatted_context=formatted_context_for_llm)

    print(f"Step 4: Calling LLM ({OLLAMA_MODEL_NAME})...")
    llm_time_start = time.time()
    llm_response_text = "LLM Call Failed or No Response"
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': full_prompt_for_llm}],
        )
        llm_response_text = response['message']['content']
        print(f"  LLM call completed in {time.time() - llm_time_start:.2f}s")
    except Exception as e:
        print(f"  Error calling LLM: {e}")
        return None, None, formatted_context_for_llm

    print("Step 5: Parsing LLM response...")
    question = None
    answer = None
    try:
        q_marker = "QUESTION:"
        a_marker = "ANSWER:"
        
        q_start_index = llm_response_text.upper().find(q_marker)
        a_start_index = llm_response_text.upper().find(a_marker)

        if q_start_index != -1 and a_start_index != -1:
            # Ensure ANSWER: comes after QUESTION:
            if a_start_index > q_start_index:
                question_content_start = q_start_index + len(q_marker)
                question = llm_response_text[question_content_start:a_start_index].strip()
                
                answer_content_start = a_start_index + len(a_marker)
                answer = llm_response_text[answer_content_start:].strip()
                print("  Successfully parsed Q&A.")
            else: 
                print("  Warning: Found Q&A markers but 'ANSWER:' appeared before 'QUESTION:'. Parsing might be incorrect.")
                print(f"  Raw LLM response:\n{llm_response_text}")

        else: 
            print("  Could not parse Q&A from LLM response using defined markers. LLM might not have followed format.")
            print(f"  Raw LLM response:\n{llm_response_text}")


    except Exception as e:
        print(f"  Error during parsing LLM response: {e}")
        print(f"  Raw LLM response:\n{llm_response_text}")

    return question, answer, formatted_context_for_llm # Return context for inspection

def generate_topic_qa_set(user_topic, embedding_model, collection, num_questions=3):
    """
    Generates a set of Q&A pairs for a given topic using the RAG pipeline.
    """
    print(f"\n--- RAG Pipeline for Topic: '{user_topic}' (Generating {num_questions} Q&As) ---")

    N_RESULTS_FOR_TOPIC = 5 
    print("Step 1: Retrieving context for topic...")
    retrieval_time_start = time.time()
    retrieved_context_raw = retrieve_relevant_chunks(
        user_topic, 
        embedding_model, 
        collection, 
        n_results=N_RESULTS_TO_RETRIEVE_FOR_RAG
    )
    print(f"  Context retrieval took {time.time() - retrieval_time_start:.2f}s")

    if not retrieved_context_raw:
        print("  Retrieval failed. Cannot generate Q&A set.")
        return None, "Retrieval failed."
    
    if retrieved_context_raw is None:
        print("  Retrieval failed or no relevant context found. Cannot generate Q&A set.")
        return [{"question": "Topic Not Found", "answer": "I could not find any information related to your topic in the textbook. Please try a different topic."}], "No relevant context retrieved."
        
    print("Step 2: Formatting context...")
    formatted_context_for_llm = format_context_for_llm(retrieved_context_raw, max_chunks_to_use=N_RESULTS_FOR_TOPIC)
    
    if "No relevant context found" in formatted_context_for_llm:
        print("  No relevant context retrieved.")
        return None, formatted_context_for_llm

    print("Step 3: Constructing LLM prompt for topic...")
    full_prompt_for_llm = TOPIC_QA_PROMPT_TEMPLATE.format(
        formatted_context=formatted_context_for_llm,
        user_topic=user_topic,
        num_questions=num_questions
    )

    print(f"Step 4: Calling LLM ({OLLAMA_MODEL_NAME})...")
    llm_time_start = time.time()
    llm_response_text = ""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': full_prompt_for_llm}],
        )
        llm_response_text = response['message']['content']
        print(f"  LLM call completed in {time.time() - llm_time_start:.2f}s")
    except Exception as e:
        print(f"  Error calling LLM: {e}")
        return None, formatted_context_for_llm

    print("Step 5: Parsing LLM response...")
    qa_pairs = parse_multiple_qa(llm_response_text, num_questions)
    
    return qa_pairs, formatted_context_for_llm 
    
if __name__ == '__main__':
    print("Running self-test for rag_core.py...")
    initialize_components()
    
    print("\n--- Testing Single Query RAG ---")
    q_test_single = "What are Type I and Type II errors?"
    question, answer, context = generate_qa_via_rag(q_test_single)
    if question and answer:
        print(f"Test Query: {q_test_single}")
        print(f"Generated Question: {question}")
        print(f"Generated Answer: {answer}")
    else:
        print(f"Single query test failed for '{q_test_single}'.")

    print("\n--- Testing Topic-Based RAG ---")
    q_test_topic = "Neyman-Pearson Lemma"
    qa_set, context = generate_topic_qa_set(q_test_topic, num_questions=2)
    if qa_set:
        print(f"Test Topic: {q_test_topic}")
        for qa in qa_set:
            print(f"  Q{qa['question_num']}: {qa['question']}")
            print(f"  A{qa['question_num']}: {qa['answer']}")
    else:
        print(f"Topic-based test failed for '{q_test_topic}'.")