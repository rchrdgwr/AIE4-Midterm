import pprint

from utilities.constants import (
    CHUNKING_STRATEGY_RECURSIVE,
    CHUNKING_STRATEGY_TABLE_AWARE,
    CHUNKING_STRATEGY_SECTION_BASED
)

class ModelRunState:
    def __init__(self):
        self.name = ""

        self.qa_model_name = "gpt-4o"
        self.qa_model = None

        self.embedding_model_name = "text-embedding-3-small"
        self.embedding_model = None

        self.chunking_strategy = CHUNKING_STRATEGY_RECURSIVE
        self.chunk_size = 1000
        self.chunk_overlap = 100

        self.response_dataset = []

        self.combined_document_objects = []
        self.retriever = None
        
        self.ragas_results = None
        self.system_template = "You are a helpful assistant"
    
    def display(self):
        pprint.pprint(self.__dict__)

    def parameters(self):
        print(f"Base model: {self.qa_model_name}")
        print(f"Embedding model: {self.embedding_model_name}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Chunk overlap: {self.chunk_overlap}")

    def results_summary(self):
        print(self.ragas_results)

    def results(self):
        results_df = self.ragas_results.to_pandas()
        results_df

    @classmethod
    def compare_ragas_results(cls, model_run_1, model_run_2):
        if not isinstance(model_run_1, cls) or not isinstance(model_run_2, cls):
            raise ValueError("Both instances must be of the same class")
