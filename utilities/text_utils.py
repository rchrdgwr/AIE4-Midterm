import os
from typing import List
import fitz # pymupdf
import tempfile
from utilities_2.text_utils import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load the file
# handle .txt and .pdf
        
class FileLoader:
    
    def __init__(self, encoding: str = "utf-8"):       
        self.documents = []
        self.encoding = encoding
        self.temp_file_path = ""


    def load_file(self, file, use_rct):
        if use_rct:
            text_splitter=MyRecursiveCharacterTextSplitter()
        else:
            text_splitter=CharacterTextSplitter()
        file_extension = os.path.splitext(file.name)[1].lower()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=file_extension) as temp_file:
            self.temp_file_path = temp_file.name
            temp_file.write(file.content)

        if os.path.isfile(self.temp_file_path):
            if self.temp_file_path.endswith(".txt"):
                self.load_text_file()
            elif self.temp_file_path.endswith(".pdf"):
                self.load_pdf_file()
            else:
                raise ValueError(
                    f"Unsupported file type: {self.temp_file_path}"
                )
            return text_splitter.split_text(self.documents)
        else:
            raise ValueError(
                    "Not a file"
                )

    def load_text_file(self):
        with open(self.temp_file_path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_pdf_file(self):
        # pymupdf
        pdf_document = fitz.open(self.temp_file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            self.documents.append(text)

class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_text(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

    

class MyRecursiveCharacterTextSplitter:
    # uses langChain.RecursiveCharacterTextSplitter
    def __init__(
        self
    ):
        self.RCTS = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, texts: List[str]) -> List[str]:
        all_chunks = []
        for doc in texts:
            chunks = self.RCTS.split_text(doc)
            all_chunks.extend(chunks)
        return all_chunks


