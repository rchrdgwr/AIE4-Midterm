from langchain_community.document_loaders import PyMuPDFLoader
import fitz
import os
import requests

from utilities.debugger import dprint
import uuid



def download_document(app_state, url, file_name, download_folder):
    file_path = os.path.join(download_folder, file_name)
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    if not os.path.exists(file_path):
        print(f"Downloading {file_name} from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            dprint(app_state, f"Failed to download document from {url}. Status code: {response.status_code}")
    else:
        dprint(app_state, f"{file_name} already exists locally.")
    return file_path

def get_documents(app_state):
    for url in app_state.document_urls:
        dprint(app_state, f"Downloading and loading document from {url}...")
        file_name = url.split("/")[-1]
        file_path = download_document(app_state, url, file_name, app_state.download_folder)
        loader = PyMuPDFLoader(file_path)
        loaded_document = loader.load()
        single_text_document = "\n".join([doc.page_content for doc in loaded_document])
        dprint(app_state, f"Number of pages: {len(loaded_document)}")
        # lets get titles and metadata
        pdf = fitz.open(file_path)
        metadata = pdf.metadata
        title = metadata.get('title', 'Document 1')

        document = {
            "url": url,
            "title": title,
            "metadata": metadata,
            "loaded_document": loaded_document,
            "single_text_document": single_text_document,
            "document_id": str(uuid.uuid4())
        }
        app_state.add_document(document)
        dprint(app_state, f"Title of Document: {title}")
        dprint(app_state, f"Full metadata for Document 1: {metadata}")
        pdf.close()
    print(f"Total documents: {len(app_state.documents)}")
