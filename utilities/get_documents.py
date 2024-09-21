import requests
import os
from langchain.document_loaders import PyMuPDFLoader

# Define the URLs for the documents
url_1 = "https://example.com/Blueprint-for-an-AI-Bill-of-Rights.pdf"
url_2 = "https://example.com/NIST.AI.600-1.pdf"

# Define local file paths for storing the downloaded PDFs
file_path_1 = "data/Blueprint-for-an-AI-Bill-of-Rights.pdf"
file_path_2 = "data/NIST.AI.600-1.pdf"

# Function to download a file from a URL
def download_pdf(url, file_path):
    # Check if the file already exists to avoid re-downloading
    if not os.path.exists(file_path):
        print(f"Downloading {file_path} from {url}...")
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"{file_path} already exists, skipping download.")
        
# Download the PDFs from the URLs
download_pdf(url_1, file_path_1)
download_pdf(url_2, file_path_2)

# Load the PDFs using PyMuPDFLoader
loader_1 = PyMuPDFLoader(file_path_1)
documents_1 = loader_1.load()

loader_2 = PyMuPDFLoader(file_path_2)
documents_2 = loader_2.load()