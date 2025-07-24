import truststore
truststore.inject_into_ssl()

import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader # Or directly use LlamaParse

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-AYL8KV3F2OGzGgchmnerCVN5AJZ1ZfqoJD9D3XuxEktgS8U6"

pdf_path = "/Users/jadhav2/Documents/prep/practice-projects/ds_textbook_qa/data/Chapter_9_Pages_559_to_628.pdf" 
target_chapter_number = None

output_markdown_filename = f"./data/chapter_{target_chapter_number}_llamaparse.md"

def parse_pdf_with_llamaparse(doc_path, result_type="markdown"):
    """
    Parses a PDF using LlamaParse.
    Note: LlamaParse typically takes a file path and processes it.
    If you need to parse specific pages of a large PDF to simulate a chapter,
    you might need to:
    1. Split the PDF into a temporary PDF containing only your chapter's pages.
    2. Or parse the whole PDF and then programmatically extract the markdown
       relevant to your chapter (which would require identifying chapter boundaries
       in the markdown output).
    """
    print(f"Starting LlamaParse for {doc_path}...")
    try:
        # Initialize the parser
        # You can specify result_type="text" or "markdown"
        # For OCR on embedded images/scans, you might need to set `parsing_instruction`
        # or other parameters related to OCR if the API supports it.
        # Check LlamaParse documentation for the latest options.
        parser = LlamaParse(
            result_type=result_type, # "markdown" or "text"
            verbose=True,
            # language="eng", # Optional: if you know the language
            # gpt4o_mode=False, # Potentially uses a faster/cheaper model if False
            # gpt4o_api_key="sk-...", # If you want to use your own OpenAI key for OCR part
            # parsing_instruction="...", # Specific instructions for parsing
        )

        # LlamaParse works with LlamaIndex's SimpleDirectoryReader or by directly calling .load_data()
        # Option 1: Using SimpleDirectoryReader (common in LlamaIndex examples)
        # This expects a directory. So, place your PDF (or the chapter PDF) in a temp directory.
        # temp_dir = "./temp_pdf_dir/"
        # os.makedirs(temp_dir, exist_ok=True)
        # shutil.copy(doc_path, os.path.join(temp_dir, os.path.basename(doc_path)))
        # reader = SimpleDirectoryReader(temp_dir, file_parser={".pdf": parser})
        # documents = reader.load_data()

        # Option 2: Directly using LlamaParse on a file path (simpler for single file)
        # Note: The exact API for LlamaParse might evolve. Check their docs.
        # This is a conceptual way it might work for a single file:
        documents = parser.load_data(doc_path) # `documents` will be a list of LlamaIndex Document objects

        print("LlamaParse finished.")
        if documents:
            # Assuming single document processed, its content is in documents[0].text
            return documents[0].text # This will be the Markdown content
        else:
            return None
    except Exception as e:
        print(f"Error during LlamaParse: {e}")
        return None

def save_text_to_file(text, filename):
    """Saves the given text to a file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"File saved successfully to {filename}.")
    except Exception as e:
        print(f"Error saving file {filename}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Using LlamaParse for PDF Chapter Extraction ---")

    # Important consideration: LlamaParse usually ingests a whole file.
    # If your PDF is the whole book, you have two main strategies:
    # 1. Pre-split PDF: Create a new PDF containing ONLY the pages of your target chapter.
    #    Then pass this smaller PDF to LlamaParse. (Recommended for focused testing)
    #    You can use PyMuPDF to create this smaller PDF.
    #
    # 2. Parse Whole Book & Extract: Parse the entire book PDF with LlamaParse.
    #    Then, you'll need to write Python code to search through the generated
    #    Markdown output to find the start and end of your target chapter and extract
    #    that portion. This requires knowing how chapters are marked in the Markdown.

    # For this example, let's assume you've pre-split the PDF to be just your chapter.
    # Or you're parsing the whole book and will manually inspect the output for your chapter.
    chapter_markdown_content = parse_pdf_with_llamaparse(pdf_path, result_type="markdown")

    if chapter_markdown_content:
        save_text_to_file(chapter_markdown_content, output_markdown_filename)
        print(f"\nMarkdown output saved to '{output_markdown_filename}'. Please inspect this file.")
        print("Next steps would be to parse this Markdown (if needed) instead of raw text.")
    else:
        print("\nLlamaParse process failed or returned no content.")