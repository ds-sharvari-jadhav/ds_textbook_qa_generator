import truststore
truststore.inject_into_ssl()

import streamlit as st
from rag_core import initialize_components, generate_qa_via_rag, generate_topic_qa_set

# This decorator ensures that heavy models are loaded only once.
@st.cache_resource
def load_resources():
    """This function loads heavy resources once and returns them."""
    print("UI: Loading resources for the first time...")
    embedding_model, collection = initialize_components()
    st.write("UI: Resources loaded successfully.")
    return embedding_model, collection

st.title("📚 AI Q&A Generator for Data Science Textbooks")
st.write("This app uses a RAG pipeline to generate interview-style questions from the 'Introduction to Probability for Data Science' textbook.")

try:
    embedding_model, collection = load_resources()
    if embedding_model is None or collection is None:
        st.error("Failed to load necessary AI components (Embedding Model or Vector DB). Please check the console logs for errors.")
        st.stop() # Stop the app from running further
except Exception as e:
    st.error(f"An error occurred during resource loading: {e}")
    st.stop()

# st.header("Choose a Mode")
# app_mode = st.radio(
#     "Select the generation mode:",
#     ("Single Question (from Query)", "Topic-Based Q&A Set")
# )

# if app_mode == "Single Question (from Query)":
#     st.subheader("Generate a Single Q&A from a Query")
#     user_query = st.text_input("Enter your query or question about a concept:", "What are the properties of maximum likelihood estimators?")

#     if st.button("Generate Single Q&A"):
#         if user_query:
#             with st.spinner("Finding context and generating Q&A... This might take a minute."):
#                 question, answer, _ = generate_qa_via_rag(user_query, embedding_model, collection)
#    
            
#             st.success("Generated Q&A:")
#             st.info(f"**Question:** {question}")
#             st.write(f"**Answer:** {answer}")
#         else:
#             st.warning("Please enter a query.")

# elif app_mode == "Topic-Based Q&A Set":
st.subheader("Generate a Study Set for a Topic")
user_topic = st.text_input("Enter the topic you want to study:", "Confidence Intervals")
num_q = st.slider("Number of questions to generate:", min_value=2, max_value=5, value=3)

if st.button("Generate Study Set"):
    if user_topic:
        with st.spinner(f"Finding context and generating {num_q} Q&As... This might take over a minute."):
            qa_set, _ = generate_topic_qa_set(user_topic, embedding_model, collection, num_questions=num_q)
        if qa_set and qa_set[0].get("question") == "Topic Not Found":
                st.error(qa_set[0].get("answer")) # Display the user-friendly error
        elif qa_set:
            st.success(f"Generated Study Set for '{user_topic}':")
            for i, qa in enumerate(qa_set):
                st.info(f"**Question {i+1}:** {qa.get('question', 'N/A')}")
                st.write(f"**Answer {i+1}:** {qa.get('answer', 'N/A')}")
                st.divider()
        else:
            st.error("Could not generate a study set for this topic. Try rephrasing.")
    else:
        st.warning("Please enter a topic.")