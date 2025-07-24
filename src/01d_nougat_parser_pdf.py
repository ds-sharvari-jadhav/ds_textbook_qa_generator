import truststore
truststore.inject_into_ssl()

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel
import os
import re
import time

chapter_pdf_path = "/Users/jadhav2/Documents/prep/practice-projects/ds_textbook_qa/data/Chapter_9_Pages_559_to_628.pdf" 

output_markdown_filename = f"./data/chapter_{os.path.basename(chapter_pdf_path).split('.')[0]}_nougat.md"

model_name = "facebook/nougat-small"

PAGES_TO_PROCESS = None # Set to None to process all pages

# --- Nougat Processing Logic ---

def process_pdf_with_nougat(pdf_path, model_name, max_pages=None):
    """
    Processes a PDF using Nougat model to generate Markdown.

    Args:
        pdf_path (str): Path to the PDF file.
        model_name (str): Name of the Nougat model on Hugging Face Hub.
        max_pages (int, optional): Maximum number of pages to process. Defaults to None (all pages).

    Returns:
        str: The generated Markdown content for the processed pages, or None if error.
    """
    print(f"Loading Nougat model: {model_name}...")
    start_load_time = time.time()
    try:
        processor = NougatProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        print(f"Model loaded in {time.time() - start_load_time:.2f} seconds.")
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        print("Ensure you have internet connection and necessary libraries installed.")
        return None

    # Device setup (try to use GPU/MPS if available, otherwise CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon GPU).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device (NVIDIA GPU).")
    else:
        device = torch.device("cpu")
        print("Using CPU device. Processing will be slow.")
    model.to(device)

    print(f"Opening PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return None

    num_pages_total = doc.page_count
    pages_to_process = num_pages_total if max_pages is None else min(max_pages, num_pages_total)
    print(f"PDF has {num_pages_total} pages. Processing {pages_to_process} pages.")

    full_markdown_content = []
    processing_times = []

    for page_num in range(pages_to_process):
        page_start_time = time.time()
        print(f"\nProcessing Page {page_num + 1} of {pages_to_process}...")
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150) # Experiment with DPI (e.g., 96, 150, 200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Process image
            print("  Preprocessing image...")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values

            # Generate markdown
            print("  Generating markdown using Nougat model...")
            gen_start_time = time.time()
            outputs = model.generate(
                pixel_values.to(device),
                min_length=1,
                max_new_tokens=4096, # Adjust based on expected page complexity / model limits
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
            )
            gen_time = time.time() - gen_start_time
            print(f"  Generation took {gen_time:.2f} seconds.")

            # Decode and clean up
            print("  Decoding output...")
            sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            sequence = processor.post_process_generation(sequence, fix_markdown=True) # Nougat specific cleanup

            full_markdown_content.append(sequence)

            page_end_time = time.time()
            page_time = page_end_time - page_start_time
            processing_times.append(page_time)
            print(f"Page {page_num + 1} processed in {page_time:.2f} seconds.")

        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")
            full_markdown_content.append(f"\n\n[ERROR PROCESSING PAGE {page_num + 1}]\n\n")

    doc.close()
    print("\n--- Processing Summary ---")
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        total_time = sum(processing_times)
        print(f"Processed {len(processing_times)} pages.")
        print(f"Average time per page: {avg_time:.2f} seconds.")
        print(f"Total processing time: {total_time:.2f} seconds.")
    else:
        print("No pages were processed successfully.")

    return "\n\n".join(full_markdown_content) # Join pages with double newline


def save_text_to_file(text, filename):
    """Saves the given text to a file."""
    try:
        print(f"Saving markdown output to {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print("File saved successfully.")
    except Exception as e:
        print(f"Error saving file {filename}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Running Nougat PDF Processor ---")
    if not os.path.exists(chapter_pdf_path):
        print(f"Error: Input PDF not found at '{chapter_pdf_path}'")
    else:
        overall_start_time = time.time()
        final_markdown = process_pdf_with_nougat(chapter_pdf_path, model_name, max_pages=PAGES_TO_PROCESS)
        overall_end_time = time.time()

        print(f"\nTotal script execution time: {(overall_end_time - overall_start_time)/60:.2f} minutes.")

        if final_markdown is not None:
            save_text_to_file(final_markdown, output_markdown_filename)
            print(f"\nProcessing complete. Output saved to '{output_markdown_filename}'.")
            print("Please carefully inspect the output file, especially formula rendering.")
        else:
            print("\nProcessing failed.")

    print("--- End of Nougat Script ---")