from utilities.constants import (
    CHUNKING_STRATEGY_TABLE_AWARE,
    CHUNKING_STRATEGY_SECTION_BASED,
    CHUNKING_STRATEGY_SEMANTIC
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
import numpy as np
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from utilities.debugger import dprint

def create_vector_store(app_state, model_run_state, **kwargs):
    for key, value in kwargs.items():
        if hasattr(model_run_state, key):
            setattr(model_run_state, key, value)
        else:
            print(f"Warning: {key} is not an attribute of the state object")

    # Rest of your create_vector_store logic
    dprint(app_state, f"Chunk size after update: {model_run_state.chunk_size}")
    create_chunked_documents(app_state, model_run_state)

    qdrant_vectorstore = Qdrant.from_documents(
        documents=model_run_state.combined_document_objects,
        embedding=model_run_state.embedding_model,
        location=":memory:" 
    )
    qdrant_retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k":app_state.num_retrievals})
    model_run_state.retriever = qdrant_retriever
    print("Vector store created")

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

def create_chunked_documents(app_state, model_run_state):
    dprint(app_state, model_run_state.chunking_strategy)
    if model_run_state.chunking_strategy == CHUNKING_STRATEGY_TABLE_AWARE:
        combined_document_objects = chunk_with_table_aware(app_state, model_run_state)
    elif model_run_state.chunking_strategy == CHUNKING_STRATEGY_SECTION_BASED:
        combined_document_objects = chunk_with_section_based(app_state, model_run_state)
    elif model_run_state.chunking_strategy == CHUNKING_STRATEGY_SEMANTIC:
        combined_document_objects = chunk_with_semantic_splitter(app_state, model_run_state)
    else:
        combined_document_objects = chunk_with_recursive_splitter(app_state, model_run_state)
    model_run_state.combined_document_objects = combined_document_objects
    dprint(app_state, "Chunking completed successfully")


def chunk_with_recursive_splitter(app_state, model_run_state):    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=model_run_state.chunk_size,  
        chunk_overlap=model_run_state.chunk_overlap,
        length_function = tiktoken_len,
    )
    combined_document_objects = []
    dprint(app_state, "Chunking documents and creating document objects")
    for document in app_state.documents:
        dprint(app_state, f"processing documend: {document['title']}")
        text = document["single_text_document"]
        dprint(app_state, text)
        title = document["title"]
        # document_id = document["document_id"]
        chunks_document = text_splitter.split_text(text)
        dprint(app_state, len(chunks_document))

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
    return combined_document_objects
    
def chunk_with_table_aware(app_state, model_run_state):
    combined_document_objects = []
    dprint(app_state, "Using Table-Aware Chunking for documents.")

    for document in app_state.documents:
        title = document["title"]
        text = document["single_text_document"]

        # Check if document is a PDF and contains tables
        if document.get("is_pdf", False):
            with pdfplumber.open(document["file_path"]) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        table_content = "\n".join([str(row) for row in table])
                        document_objects = Document(
                            page_content=table_content,
                            metadata={
                                "source": title,
                                "document_id": document.get("document_id", "default_id"),
                                "chunk_number": "table"
                            }
                        )
                        combined_document_objects.append(document_objects)
        
        # Chunk the rest of the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=model_run_state.chunk_size, chunk_overlap=model_run_state.chunk_overlap)
        chunks_document = text_splitter.split_text(text)

        for chunk_number, chunk in enumerate(chunks_document, start=1):
            document_objects = Document(
                page_content=chunk,
                metadata={
                    "source": title,
                    "document_id": document.get("document_id", "default_id"),
                    "chunk_number": chunk_number
                }
            )
            combined_document_objects.append(document_objects)

    return combined_document_objects


def chunk_with_section_based(app_state, model_run_state):
    combined_document_objects = []
    dprint(app_state, "Using Section-Based Chunking for documents.")

    for document in app_state.documents:
        text = document["single_text_document"]
        title = document["title"]

        # Split the text by headings
        sections = re.split(r"\n[A-Z].+?\n", text)

        # Chunk each section
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=model_run_state.chunk_size, chunk_overlap=model_run_state.chunk_overlap)
        for section_number, section in enumerate(sections, start=1):
            chunks_document = text_splitter.split_text(section)
            for chunk_number, chunk in enumerate(chunks_document, start=1):
                document_objects = Document(
                    page_content=chunk,
                    metadata={
                        "source": title,
                        "document_id": document.get("document_id", "default_id"),
                        "section_number": section_number,
                        "chunk_number": chunk_number
                    }
                )
                combined_document_objects.append(document_objects)

    return combined_document_objects


def chunk_with_semantic_splitter(app_state, model_run_state):
    # Load pre-trained model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    combined_document_objects = []
    dprint(app_state, "Using Semantic-Based Chunking for documents.")

    for document in app_state.documents:
        text = document["single_text_document"]
        title = document["title"]

        # Split text into sentences or paragraphs
        sentences = text.split(". ")  # Simple split by sentence (you can refine this)
        sentence_embeddings = model.encode(sentences)

        # Group sentences into chunks based on semantic similarity
        chunks = []
        current_chunk = []
        for i in range(len(sentences) - 1):
            current_chunk.append(sentences[i])
            # Calculate similarity between consecutive sentences
            sim = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
            if sim < 0.7 or len(current_chunk) >= model_run_state.chunk_size:
                # If similarity is below threshold or chunk size is reached, start a new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        # Add the final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Create document objects for the chunks
        for chunk_number, chunk in enumerate(chunks, start=1):
            document_objects = Document(
                page_content=chunk,
                metadata={
                    "source": title,
                    "document_id": document.get("document_id", "default_id"),
                    "chunk_number": chunk_number
                }
            )
            combined_document_objects.append(document_objects)

    return combined_document_objects