import pprint
class AppState:
    def __init__(self):
        self.debug = False
        self.document_urls = []
        self.download_folder = "data/"
        self.documents = []

    def display(self):
        pprint.pprint(self.__dict__)
    def set_document_urls(self, document_urls):
        self.document_urls = document_urls
    def add_document(self, document):
        self.documents.append(document)
    def set_debug(self, debug):
        self.debug = debug