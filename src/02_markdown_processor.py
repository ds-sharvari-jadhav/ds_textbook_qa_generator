from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import json
import os

def load_markdown_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Successfully loaded: {filepath}")
        return content
    except FileNotFoundError:
        print(f"Error: Markdown file not found at {filepath}")
        return None

if __name__ == "__main__":
    nougat_md_path = "./data/chapter_Chapter_9_Pages_559_to_628_nougat.md"
    markdown_content = load_markdown_file(nougat_md_path)

    output_chunk_filename = "./data/chapter_9_nougat_chunks_with_metadata.json"
    
    # Split by major headers
    headers_to_split_on = [
        ("##", "L1_Header"), # Chapter titles
        ("###", "L2_Header"), # Section titles
        ("####", "L3_Header") # Subsection titles
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    header_splits = markdown_splitter.split_text(markdown_content)
    
    final_chunks_with_metadata = []
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
    )
    
    for doc in header_splits:
        current_content = doc.page_content
        current_metadata = doc.metadata.copy()
    
        sub_chunks = char_splitter.split_text(current_content)
    
        for i, sub_chunk_text in enumerate(sub_chunks):
            chunk_metadata = current_metadata.copy()
            chunk_metadata["chunk_index_within_section"] = i
    
            final_chunks_with_metadata.append({
                "text": sub_chunk_text,
                "metadata": chunk_metadata
            })
    
    # Print some results to inspect
    print(f"Total chunks created: {len(final_chunks_with_metadata)}")
    for i in range(min(5, len(final_chunks_with_metadata))):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Metadata: {final_chunks_with_metadata[i]['metadata']}")
        print(f"Text: {final_chunks_with_metadata[i]['text'][:300]}...") 

    try:
        print(f"\nSaving {len(final_chunks_with_metadata)} chunks to '{output_chunk_filename}'...")
        with open(output_chunk_filename, 'w', encoding='utf-8') as f:
            json.dump(final_chunks_with_metadata, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved chunks to '{output_chunk_filename}'")
    except Exception as e:
        print(f"Error saving chunks to JSON file '{output_chunk_filename}': {e}")
