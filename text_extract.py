import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def get_pdf_text(folder_path):
    text = ""
    # Get all PDF files in the folder
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    # Iterate over each PDF file
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            print(f"Error reading {pdf}: {e}")
    return text

# Save the extracted text into a .txt file
def save_text_to_file(text, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")

def read_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading from {file_path}: {e}")
        return ""

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150, length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Specify paths for PDF folder and output text file
folder_path = '/Users/manish/Downloads/DiagnoAI-main/data'  # Change this if your PDFs are in a different directory
output_file = '/Users/manish/Downloads/DiagnoAI-main/raw_text.txt'

# Extract text from PDFs and save to raw_text.txt if it doesn’t exist
if not os.path.exists(output_file):
    raw_text = get_pdf_text(folder_path)
    save_text_to_file(raw_text, output_file)

# Read the text from raw_text.txt and split into chunks
raw_text = read_text_from_file(output_file)
if raw_text:
    chunks = get_text_chunks(raw_text)
    # Optional: Print the chunks to verify
    print(chunks)
else:
    print("No text found to process.")
