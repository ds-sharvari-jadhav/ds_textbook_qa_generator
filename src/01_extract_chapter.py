import fitz  # PyMuPDF
import re
import os

pdf_path = "/Users/jadhav2/Documents/prep/datascience_chan.pdf" 
target_chapter_number = 9
start_page_index = 558 #542
end_page_index = 627 #611

# output_raw_filename = f"./data/chapter_{target_chapter_number}_raw.txt"
# output_cleaned_filename = f"./data/chapter_{target_chapter_number}_cleaned.txt"

output_raw_filename = f"./data/chapter_{target_chapter_number}_raw_with_tags.txt"
output_cleaned_filename = f"./data/chapter_{target_chapter_number}_cleaned_with_tags.txt"

def extract_text_from_chapter(pdf_path, start_page, end_page):
    """
    Extracts text from a specific page range in a PDF.

    Args:
        pdf_path (str): The file path to the PDF document.
        start_page (int): The 0-based index of the starting page.
        end_page (int): The 0-based index of the ending page (inclusive).

    Returns:
        str: The concatenated text extracted from the specified pages,
             or None if an error occurs.
    """
    chapter_text = ""
    try:
        print(f"Opening PDF: {pdf_path}")
        document = fitz.open(pdf_path)

        # Validate page range
        num_pages_total = document.page_count
        if start_page < 0 or end_page >= num_pages_total or start_page > end_page:
            print(f"Error: Invalid page range. Start: {start_page}, End: {end_page}, Total Pages: {num_pages_total}")
            document.close()
            return None

        print(f"Extracting text from page {start_page + 1} to {end_page + 1}...")

        # Loop through the specified page range (inclusive of end_page)
        for page_num in range(start_page, end_page + 1):
            print(f"  Processing page index: {page_num} (Visual Page: {page_num + 1})")
            page = document.load_page(page_num)
            page_text = page.get_text("xhtml") # page.get_text("text") 
            chapter_text += page_text + "\n" 

        document.close()
        print("Finished extracting text.")
        return chapter_text

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        if 'document' in locals() and document:
            document.close()
        return None

def clean_text(text):
    """
    Performs basic text cleaning.

    Args:
        text (str): The raw text to clean.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""

    print("Cleaning text...")
    # Remove leading/trailing whitespace from the whole text
    cleaned = text.strip()

    # Replace multiple consecutive spaces with a single space
    cleaned = re.sub(r' +', ' ', cleaned)

    # Replace multiple consecutive newlines with a double newline (paragraph break)
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)

    # Optional: Remove page numbers if they follow a consistent pattern and interfere.
    # This requires inspecting the raw text first. Example (adjust regex as needed):
    # cleaned = re.sub(r'\n\n\d+\n\n', '\n\n', cleaned) # If page numbers are on their own lines

    # Optional: Remove specific repeating headers/footers if identified.
    # Example: (Needs pattern specific to your PDF)
    # cleaned = re.sub(r'Chapter \d+:.*\n', '', cleaned) # Simple header removal example

    print("Finished cleaning text.")
    return cleaned

def save_text_to_file(text, filename):
    """Saves the given text to a file."""
    try:
        print(f"Saving text to {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print("File saved successfully.")
    except Exception as e:
        print(f"Error saving file {filename}: {e}")

if __name__ == "__main__":
    print("--- Starting Chapter Text Extraction ---")

    # 1. Extract Raw Text
    raw_text = extract_text_from_chapter(pdf_path, start_page_index, end_page_index)

    if raw_text:
        # 2. Save Raw Text (Optional but good for debugging)
        save_text_to_file(raw_text, output_raw_filename)

        # 3. Clean Text
        cleaned_text = clean_text(raw_text)

        # 4. Save Cleaned Text
        save_text_to_file(cleaned_text, output_cleaned_filename)
        print(f"\nProcess complete. Cleaned text saved to '{output_cleaned_filename}'")
    else:
        print("\nProcess failed. No text extracted or an error occurred.")

    print("--- Finished extraction and saving files ---")
