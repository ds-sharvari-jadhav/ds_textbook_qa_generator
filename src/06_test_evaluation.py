import truststore
truststore.inject_into_ssl()

import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import time
import json
import os
import re

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_DB_PATH = "./chroma_db_nougat_chapter9"
CHROMA_COLLECTION_NAME = "chapter9_nougat_content_v1"
N_RESULTS_TO_RETRIEVE_FOR_RAG = 3
OLLAMA_MODEL_NAME = 'phi3:mini-4k'

def retrieve_relevant_chunks(query_text, n_results=N_RESULTS_TO_RETRIEVE_FOR_RAG):
    """
    Retrieves relevant text chunks from the globally defined ChromaDB collection.
    """
    # print(f"Retrieving for: {query_text}") # Optional debug print
    try:
        query_embedding_array = embedding_model_global.encode([query_text])
        query_embedding_list = query_embedding_array[0].tolist()
        
        results = collection_global.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results,
            include=['documents', 'metadatas'] # Only need these for context
        )
        return results
    except Exception as e:
        print(f"Error during retrieval for query '{query_text}': {e}")
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

LLM_PROMPT_TEMPLATE = """
You are an expert AI assistant for Data Science interview preparation. Your task is to generate one insightful, interview-style question and a concise, accurate answer based *strictly* on the provided textbook context.

{formatted_context}

Based *only* on the context provided above, perform the following:
1. Generate one insightful, interview-style question that tests understanding of a key concept. **The question must be a general, standalone question and MUST NOT refer to "the text", "the document", "the provided context", or any similar phrasing.**
2. Provide a clear and concise answer to that question. **CRITICAL INSTRUCTION: You MUST NOT use any information outside of the provided context snippets.**

Format your response *exactly* as follows, with no preamble or extra text:
QUESTION:
[Your generated question here]
ANSWER:
[Your generated answer here]
"""
TOPIC_QA_PROMPT_TEMPLATE = """
You are an expert AI assistant for Data Science interview preparation. A user wants to study a specific topic, and your task is to generate a set of challenging interview questions and their concise answers based *strictly* on the provided context.

{formatted_context}

The user is interested in the topic: "{user_topic}".

Based *only* on the context provided above, generate {num_questions} distinct interview-style questions related to this topic.

**IMPORTANT RULES FOR QUESTION GENERATION:**
1.  **Standalone Questions:** Each question must be a general, standalone question. It **MUST NOT** refer to "the text", "the document", "the provided context", "according to the excerpt", or any similar phrasing that reveals it's based on a given text. It should sound like a natural interview question.
2.  **Grounded Answers:** Each answer MUST be derived solely from the provided context snippets. Do not add outside information. If the context is insufficient, state so in the answer.

Format your response *exactly* as follows. Do not add any extra formatting, styling, or text.
Here is a perfect example of the required format:
Q1: What is the primary trade-off managed by the Neyman-Pearson lemma?
A1: The primary trade-off is between maximizing the detection rate (power) while maintaining a fixed false alarm rate (Type I error).

Now, using the context I provided, generate your own set of questions and answers in that exact format:

Q1: [Your generated question 1 here]
A1: [Your generated answer 1 here]

Q2: [Your generated question 2 here]
A2: [Your generated answer 2 here]

Q3: [Your generated question 3 here]
A3: [Your generated answer 3 here]
... (up to Q{num_questions}/A{num_questions})
"""
def generate_qa_via_rag(user_query):
    """
    Generates a Question and Answer pair using the RAG pipeline.
    Uses globally defined components: retrieve_relevant_chunks, LLM_PROMPT_TEMPLATE, OLLAMA_MODEL_NAME.
    """
    print(f"\n--- RAG Pipeline for Query: '{user_query}' ---")

    # 1. Retrieve relevant context
    print("Step 1: Retrieving context...")
    retrieval_time_start = time.time()
    retrieved_context_raw = retrieve_relevant_chunks(user_query)
    print(f"  Context retrieval took {time.time() - retrieval_time_start:.2f}s")

    if not retrieved_context_raw:
        print("  Retrieval failed. Cannot generate Q&A.")
        return None, None, "Retrieval failed."

    # 2. Format context for LLM
    print("Step 2: Formatting context...")
    formatted_context_for_llm = format_context_for_llm(retrieved_context_raw)
    
    if "No relevant context found" in formatted_context_for_llm:
        print("  No relevant context was found by retrieval. Cannot effectively prompt LLM.")
        return None, None, formatted_context_for_llm

    # 3. Construct full prompt
    print("Step 3: Constructing LLM prompt...")
    full_prompt_for_llm = LLM_PROMPT_TEMPLATE.format(formatted_context=formatted_context_for_llm)

    # 4. Call LLM
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

    # 5. Parse LLM response
    print("Step 5: Parsing LLM response...")
    question = None
    answer = None
    try:
        q_marker = "QUESTION:"
        a_marker = "ANSWER:"
        
        q_start_index = llm_response_text.upper().find(q_marker) # Case-insensitive find
        a_start_index = llm_response_text.upper().find(a_marker)

        if q_start_index != -1 and a_start_index != -1:
            # Ensure ANSWER: comes after QUESTION:
            if a_start_index > q_start_index:
                question_content_start = q_start_index + len(q_marker)
                question = llm_response_text[question_content_start:a_start_index].strip()
                
                answer_content_start = a_start_index + len(a_marker)
                answer = llm_response_text[answer_content_start:].strip()
                print("  Successfully parsed Q&A.")
            else: # Markers found but in wrong order
                print("  Warning: Found Q&A markers but 'ANSWER:' appeared before 'QUESTION:'. Parsing might be incorrect.")
                print(f"  Raw LLM response:\n{llm_response_text}")

        else: # One or both markers not found
            print("  Could not parse Q&A from LLM response using defined markers. LLM might not have followed format.")
            print(f"  Raw LLM response:\n{llm_response_text}")
            # As a fallback, you could assign the whole response to 'answer' if no markers found,
            # or try more advanced regex if needed, but for now, this is fine.

    except Exception as e:
        print(f"  Error during parsing LLM response: {e}")
        print(f"  Raw LLM response:\n{llm_response_text}")

    return question, answer, formatted_context_for_llm # Return context for inspection
    
# Assume retrieve_relevant_chunks, format_context_for_llm, OLLAMA_MODEL_NAME are defined globally or imported.

def generate_topic_qa_set(user_topic, num_questions=3):
    """
    Generates a set of Q&A pairs for a given topic using the RAG pipeline.
    """
    print(f"\n--- RAG Pipeline for Topic: '{user_topic}' (Generating {num_questions} Q&As) ---")

    # 1. Retrieve relevant context
    # We might want to retrieve slightly more chunks (e.g., 4-5) to cover the topic broadly
    N_RESULTS_FOR_TOPIC = 5 
    print("Step 1: Retrieving context for topic...")
    retrieval_time_start = time.time()
    retrieved_context_raw = retrieve_relevant_chunks(user_topic, n_results=N_RESULTS_FOR_TOPIC)
    print(f"  Context retrieval took {time.time() - retrieval_time_start:.2f}s")

    if not retrieved_context_raw:
        print("  Retrieval failed. Cannot generate Q&A set.")
        return None, "Retrieval failed."

    # 2. Format context for LLM
    print("Step 2: Formatting context...")
    formatted_context_for_llm = format_context_for_llm(retrieved_context_raw, max_chunks_to_use=N_RESULTS_FOR_TOPIC)
    
    if "No relevant context found" in formatted_context_for_llm:
        print("  No relevant context retrieved.")
        return None, formatted_context_for_llm

    # 3. Construct full prompt using the TOPIC template
    print("Step 3: Constructing LLM prompt for topic...")
    full_prompt_for_llm = TOPIC_QA_PROMPT_TEMPLATE.format(
        formatted_context=formatted_context_for_llm,
        user_topic=user_topic,
        num_questions=num_questions
    )

    # 4. Call LLM
    print(f"Step 4: Calling LLM ({OLLAMA_MODEL_NAME})...")
    llm_time_start = time.time()
    llm_response_text = ""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': full_prompt_for_llm}],
            # options={'temperature': 0.4} # Slightly higher temperature might encourage diverse questions
        )
        llm_response_text = response['message']['content']
        print(f"  LLM call completed in {time.time() - llm_time_start:.2f}s")
    except Exception as e:
        print(f"  Error calling LLM: {e}")
        return None, formatted_context_for_llm

    # 5. Parse LLM response (More complex parsing needed for multiple Q&As)
    print("Step 5: Parsing LLM response...")
    qa_pairs = parse_multiple_qa(llm_response_text, num_questions)
    
    return qa_pairs, formatted_context_for_llm 

def parse_multiple_qa(llm_response_text, expected_count):
    """
    Parses the LLM response text to extract multiple Q&A pairs (Q1/A1, Q2/A2...).
    """
    qa_list = []
    # Use regex to find patterns like Q1: ... A1: ... Q2: ...
    # This regex looks for Q[number]: followed by content, then A[number]: followed by content.
    # It handles potential newlines within the question or answer content.
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
    
print("--- RAG Pipeline Integration & First Q&A ---")

print(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
try:
    embedding_model_global = SentenceTransformer(EMBEDDING_MODEL_NAME) # Use a distinct name
except Exception as e:
    print(f"Error loading sentence transformer model: {e}"); exit()
print("Sentence transformer model loaded.")

print(f"\nConnecting to ChromaDB at path: {CHROMA_DB_PATH}")
try:
    client_global = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection_global = client_global.get_collection(name=CHROMA_COLLECTION_NAME)
    if collection_global.count() == 0:
        print(f"Warning: ChromaDB collection '{CHROMA_COLLECTION_NAME}' is empty!")
except Exception as e:
    print(f"Error connecting to ChromaDB or getting collection: {e}"); exit()
print(f"Connected to collection '{CHROMA_COLLECTION_NAME}'. Count: {collection_global.count()}")

try:
    ollama.list()
    print(f"\nOllama server is responsive. Using model: {OLLAMA_MODEL_NAME} for generation.")
except Exception as e:
    print(f"Error: Ollama server not responding. Ensure Ollama is running. Error: {e}"); exit()

def run_evaluation_suite():
    # single_queries = [
    #     "What is a confidence interval?",
    #     "Explain the concept of a pivotal quantity.",
    #     "Describe Type I and Type II errors."
    #     # ... add more
    # ]
    
    topics_for_generation = [
        "Maximum Likelihood Estimation",
        "Neyman-Pearson Lemma",
        "Student's t-distribution"
        # ... add more
    ]
    
    evaluation_results = []

    # print("--- Running Single Query Evaluations ---")
    # for query in single_queries:
    #     question, answer, context = generate_qa_via_rag(query)
    #     evaluation_results.append({
    #         "type": "single_query",
    #         "input": query,
    #         "context": context,
    #         "generated_question": question,
    #         "generated_answer": answer
    #     })

    print("\n--- Running Topic-Based Q&A Evaluations ---")
    for topic in topics_for_generation:
        qa_set, context = generate_topic_qa_set(topic, num_questions=3)
        evaluation_results.append({
            "type": "topic_generation",
            "input": topic,
            "context": context,
            "generated_qa_set": qa_set
        })

    # Save the results to a file
    with open("./data/evaluation_run_v2.json", 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    
    print("\nEvaluation suite finished. Results saved to 'evaluation_run_v2.json'")

if __name__ == "__main__":
    run_evaluation_suite()