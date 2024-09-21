from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
import fitz
import io
import os
import requests
import tiktoken
from utilities.debugger import dprint
import uuid

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

def download_document(state, url, file_name, download_folder):
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
            dprint(state, f"Failed to download document from {url}. Status code: {response.status_code}")
    else:
        dprint(state, f"{file_name} already exists locally.")
    return file_path

def get_documents(state):
    for url in state.document_urls:
        dprint(state, f"Downloading and loading document from {url}...")
        file_name = url.split("/")[-1]
        file_path = download_document(state, url, file_name, state.download_folder)
        loader = PyMuPDFLoader(file_path)
        loaded_document = loader.load()
        single_text_document = "\n".join([doc.page_content for doc in loaded_document])
        #state.add_loaded_document(loaded_document)  # Append the loaded documents to the list
        #state.add_single_text_document(single_text_document)
        dprint(state, f"Number of pages: {len(loaded_document)}")
        # lets get titles and metadata
        pdf = fitz.open(file_path)
        metadata = pdf.metadata
        title = metadata.get('title', 'Document 1')
        #state.add_metadata(metadata)
        #state.add_title(title)
        document = {
            "url": url,
            "title": title,
            "metadata": metadata,
            "single_text_document": single_text_document,
            "document_id": str(uuid.uuid4())
        }
        state.add_document(document)
        dprint(state, f"Title of Document: {title}")
        dprint(state, f"Full metadata for Document 1: {metadata}")
        pdf.close()
    dprint(state, f"documents: {state.documents}")

def create_chunked_documents(state):
    get_documents(state)


    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=state.chunk_size,  
        chunk_overlap=state.chunk_overlap,
        length_function = tiktoken_len,
    )
    combined_document_objects = []
    dprint(state, "Chunking documents and creating document objects")
    for document in state.documents:
        dprint(state, f"processing documend: {document['title']}")
        text = document["single_text_document"]
        dprint(state, text)
        title = document["title"]
        document_id = document["document_id"]
        chunks_document = text_splitter.split_text(text)
        dprint(state, len(chunks_document))

        for chunk_number, chunk in enumerate(chunks_document, start=1):
            document_objects = Document(
                page_content=chunk,
                metadata={
                    "source": title,
                    "document_id": document.get("document_id", "default_id"),
                    "chunk_number": chunk_number  # Add unique chunk number
                }
            )
            combined_document_objects.append(document_objects)
    state.add_combined_document_objects(combined_document_objects)
    

def create_vector_store(state, **kwargs):
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)
        else:
            print(f"Warning: {key} is not an attribute of the state object")

    # Rest of your create_vector_store logic
    print(f"Chunk size after update: {state.chunk_size}")




    create_chunked_documents(state)
    embedding_model = OpenAIEmbeddings(model=state.embedding_model)

    qdrant_vectorstore = Qdrant.from_documents(
        documents=state.combined_document_objects,
        embedding=embedding_model,
        location=":memory:" 
    )
    qdrant_retriever = qdrant_vectorstore.as_retriever()
    state.set_retriever(qdrant_retriever)
    print("Vector store created")
    return qdrant_retriever