class AppState:
    def __init__(self):
        self.debug = False
        self.llm_model = "gpt-3.5-turbo"
        self.embedding_model = "text-embedding-3-small"
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.document_urls = []
        self.download_folder = "data/"
        self.loaded_documents = []
        self.single_text_documents = []
        self.metadata = []
        self.titles = []
        self.documents = []
        self.combined_document_objects = []
        self.retriever = None

        self.system_template = "You are a helpful assistant"
        #
        self.user_input = None
        self.retrieved_documents = []
        self.chat_history = []
        self.current_question = None
    
    def set_document_urls(self, document_urls):
        self.document_urls = document_urls

    def set_llm_model(self, llm_model):
        self.llm_model = llm_model

    def set_embedding_model(self, embedding_model):
        self.embedding_model = embedding_model

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size

    def set_chunk_overlap(self, chunk_overlap):
        self.chunk_overlap = chunk_overlap

    def set_system_template(self, system_template):
        self.system_template = system_template

    def add_loaded_document(self, loaded_document):
        self.loaded_documents.append(loaded_document)

    def add_single_text_documents(self, single_text_document):
        self.single_text_documents.append(single_text_document)
    def add_metadata(self, metadata):
        self.metadata = metadata

    def add_title(self, title):
        self.titles.append(title)
    def add_document(self, document):
        self.documents.append(document)
    def add_combined_document_objects(self, combined_document_objects):
        self.combined_document_objects = combined_document_objects
    def set_retriever(self, retriever):
        self.retriever = retriever
    #
    # Method to update the user input
    def set_user_input(self, input_text):
        self.user_input = input_text

    # Method to add a retrieved document
    # def add_document(self, document):
    #     print("adding document")
    #     print(self)
    #     self.retrieved_documents.append(document)

    # Method to update chat history
    def update_chat_history(self, message):
        self.chat_history.append(message)

    # Method to get the current state
    def get_state(self):
        return {
            "user_input": self.user_input,
            "retrieved_documents": self.retrieved_documents,
            "chat_history": self.chat_history,
            "current_question": self.current_question
        }