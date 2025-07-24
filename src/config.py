# --- Configuration File ---

# RAG Pipeline Settings
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_DB_PATH = "./chroma_db_nougat_chapter9"
CHROMA_COLLECTION_NAME = "chapter9_nougat_content_v1"
N_RESULTS_TO_RETRIEVE_FOR_RAG = 5
OLLAMA_MODEL_NAME = 'phi3:mini-4k'
RELEVANCE_THRESHOLD = 0.5

# --- Prompt Templates ---
LLM_PROMPT_TEMPLATE = """
You are an expert AI assistant for Data Science interview preparation. Your task is to generate one insightful, interview-style question and a concise, accurate answer based *strictly* on the provided textbook context.

{formatted_context}

Based *only* on the context provided above, perform the following:
1. Generate one insightful, interview-style question that tests understanding of a key concept. **The question must be a general, standalone question and MUST NOT refer to "the text", "the document", "the provided context", or any similar phrasing.**
2. Provide a clear and concise answer to that question. **CRITICAL INSTRUCTION: You MUST NOT use any information outside of the provided context snippets.**

Format your response *exactly* as follows, with no preamble or extra text. The question marker must be 'Q' followed by the number:
QUESTION:
[Your generated question here]
ANSWER:
[Your generated answer here]
"""

TOPIC_QA_PROMPT_TEMPLATE = """
[SYSTEM]
You are a silent, automated Q&A generation tool. Your only purpose is to take the provided context and topic and output a structured set of questions and answers. You will not add any commentary, apologies, or explanations. You will adhere to the user's requested number of questions precisely.

[CONTEXT]
{formatted_context}

[TASK]
The user is interested in the topic: "{user_topic}".

Based *only* on the provided [CONTEXT], generate exactly {num_questions} distinct, interview-style questions and their corresponding answers.

**CRITICAL RULES:**
1.  **GROUNDED ANSWERS:** Every answer MUST be derived exclusively from the information within the [CONTEXT]. Do not use any external knowledge.
2.  **STANDALONE QUESTIONS:** The questions must be about the technical concepts themselves.
    - **DO NOT** mention the source document, the context, or the summary.
    - **DO NOT** use phrases like "according to the text", "based on the document", or refer to "Document Source". The questions must be completely independent.

**EXAMPLE OF BAD QUESTION STYLE (DO NOT DO THIS):**
- According to Document Source [9.6 Summary], what is a confidence interval?
- Based on the provided context, explain the likelihood ratio.

**EXAMPLE OF GOOD QUESTION STYLE (FOLLOW THIS STYLE):**
- What is the fundamental purpose of a confidence interval?
- What are two common misconceptions when interpreting a 95% confidence interval?

**OUTPUT FORMAT:**
Your entire response must be ONLY the Q&A pairs, formatted exactly as shown below. Do not include any other text.
Q1: [Generated question 1]
A1: [Generated answer 1]
... (and so on)
"""