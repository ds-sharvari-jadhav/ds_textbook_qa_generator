import fitz  # PyMuPDF
import os

original_pdf_path = pdf_path = "/Users/jadhav2/Documents/prep/datascience_chan.pdf" 

target_chapter_number = 9
start_page_index = 558
end_page_index = 627 

chapter_pdf_filename = f"./data/Chapter_{target_chapter_number}_Pages_{start_page_index+1}_to_{end_page_index+1}.pdf"

# --- Main PDF Splitting Logic ---
def split_pdf_to_chapter(original_pdf_path, start_idx, end_idx, output_path):
    """
    Extracts a range of pages from an original PDF and saves them as a new PDF.

    Args:
        original_pdf_path (str): Path to the original PDF file.
        start_idx (int): 0-based index of the starting page.
        end_idx (int): 0-based index of the ending page (inclusive).
        output_path (str): Path where the new chapter PDF will be saved.
    """
    try:
        print(f"Opening original PDF: {original_pdf_path}")
        original_doc = fitz.open(original_pdf_path)
    except Exception as e:
        print(f"Error: Could not open original PDF '{original_pdf_path}'. Exception: {e}")
        return False

    num_pages_total = original_doc.page_count

    # Validate page range
    if not (0 <= start_idx < num_pages_total and 0 <= end_idx < num_pages_total and start_idx <= end_idx):
        print(f"Error: Invalid page range. Start Index: {start_idx}, End Index: {end_idx}, Total Pages in PDF: {num_pages_total}")
        print("Please ensure indices are 0-based and within the PDF's bounds.")
        original_doc.close()
        return False

    print(f"Original PDF has {num_pages_total} pages.")
    print(f"Extracting pages from index {start_idx} (visual page {start_idx + 1}) to {end_idx} (visual page {end_idx + 1}).")

    # Create a new PDF document
    new_doc = fitz.open() # Creates a new empty PDF

    try:
        new_doc.insert_pdf(original_doc, from_page=start_idx, to_page=end_idx, show_progress=1)
        # Note: insert_pdf with from_page/to_page can sometimes have issues with complex PDFs or links.
        # An alternative, more robust way page-by-page (slower but often safer for structure):
        # for page_num in range(start_idx, end_idx + 1):
        #     new_doc.insert_page(new_doc.page_count, original_doc, page_num)
    except Exception as e:
        print(f"Error during page insertion: {e}")
        original_doc.close()
        new_doc.close()
        return False


    try:
        print(f"Saving chapter PDF to: {output_path}")
        new_doc.save(output_path, garbage=4, deflate=True) # Save with optimization
        print("Chapter PDF saved successfully.")
        return True
    except Exception as e:
        print(f"Error: Could not save the new chapter PDF '{output_path}'. Exception: {e}")
        return False
    finally:
        original_doc.close()
        new_doc.close()

# --- Main Execution ---
if __name__ == "__main__":
    print("--- PDF Chapter Splitter ---")

    success = split_pdf_to_chapter(original_pdf_path, start_page_index, end_page_index, chapter_pdf_filename)

    if success:
        print(f"\nSuccessfully created PDF for Chapter {target_chapter_number} at '{chapter_pdf_filename}'")
        print("You can now use this smaller PDF with LlamaParse or other processing tools.")
    else:
        print("\nFailed to create the chapter PDF.")

    print("--- End of PDF Chapter Splitter Script ---")