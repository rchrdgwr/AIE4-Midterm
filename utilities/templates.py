
from langchain_core.prompts import ChatPromptTemplate
def get_qa_prompt():

    system_template = """
        You are an expert at explaining technical documents to people.
        You are provided context below to answer the question.
        Only use the information provided below.
        If they do not ask a question, have a conversation with them and ask them if they have any questions
        If you cannot answer the question with the content below say 'I don't have enough information, sorry'
        The two documents are 'Blueprint for an AI Bill of Rights' and 'Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile'
    """
    human_template = """ 
    ===
    question:
    {question}

    ===
    context:
    {context}
    ===
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    return chat_prompt