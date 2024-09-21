import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader

from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings

import tiktoken



def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

def create_chunked_documents():
    file_path_1 = "data/Blueprint-for-an-AI-Bill-of-Rights.pdf"
    file_path_2 = "data/NIST.AI.600-1.pdf"
    loader = PyMuPDFLoader(file_path_1)
    documents_1 = loader.load()
    loader = PyMuPDFLoader(file_path_2)
    documents_2 = loader.load()
    print(f"Number of pages in 1: {len(documents_1)}")
    print(f"Number of pages in 2: {len(documents_2)}")

    text1 = "\n".join([doc.page_content for doc in documents_1])
    text2 = "\n".join([doc.page_content for doc in documents_2])

    pdf_1 = fitz.open(file_path_1)
    pdf_2 = fitz.open(file_path_2)

    # Extract metadata
    metadata_1 = pdf_1.metadata
    title_1 = metadata_1.get('title', 'Document 1')
    metadata_2 = pdf_2.metadata
    title_2 = metadata_2.get('title', 'Document 2')

    # Print the title of each document
    print(f"Title of Document 1: {title_1}")
    print(f"Title of Document 2: {title_2}")

    # Optionally, you can also access other metadata
    print(f"Full metadata for Document 1: {metadata_1}")
    print(f"Full metadata for Document 2: {metadata_2}")

    # Close the PDFs
    pdf_1.close()
    pdf_2.close()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=100,
        length_function = tiktoken_len,
    )
    chunks_document_1 = text_splitter.split_text(text1)
    chunks_document_2 = text_splitter.split_text(text2)
    document_objects_1 = [Document(page_content=chunk, metadata={"source": title_1, "document_id": "doc1"}) for chunk in chunks_document_1]

    document_objects_2 = [Document(page_content=chunk, metadata={"source": title_2, "document_id": "doc2"}) for chunk in chunks_document_1]
    combined_document_objects = document_objects_1 + document_objects_2
    print(f"Number of chunks for Document 1: {len(chunks_document_1)}")
    print(f"Number of chunks for Document 2: {len(chunks_document_2)}")
    return combined_document_objects

def create_vector_store():
    docs = create_chunked_documents()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    qdrant_vectorstore = Qdrant.from_documents(
        documents=docs,
        embedding=embedding_model,
        location=":memory:" 
    )
    qdrant_retriever = qdrant_vectorstore.as_retriever()
    return qdrant_retriever