import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utilities.rag_utilities import create_vector_store
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from classes.app_state import AppState

document_urls = [
    "https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf",
     "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
]

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup our state
state = AppState()
state.set_document_urls(document_urls)
state.set_llm_model("gpt-3.5-turbo")
state.set_embedding_model("text-embedding-3-small")


# Initialize the OpenAI LLM using LangChain
llm = ChatOpenAI(model=state.llm_model, openai_api_key=openai_api_key)




qdrant_retriever = create_vector_store(state)

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
# create the chain
openai_chat_model = ChatOpenAI(model="gpt-4o")



retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))


    | {"response": chat_prompt | openai_chat_model, "context": itemgetter("context")}
)

opening_content = """
Welcome! I can answer your questions on AI based on the following 2 documents:
- Blueprint for an AI Bill of Rights
- Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile

What questions do you have for me?
"""

@cl.on_chat_start
async def on_chat_start():

    await cl.Message(content=opening_content).send()



@cl.on_message
async def main(message):

    # formatted_prompt = prompt.format(question=message.content)

    # Call the LLM with the formatted prompt
    # response = llm.invoke(formatted_prompt)
    # 
    response = retrieval_augmented_qa_chain.invoke({"question" : message.content })
    answer_content = response["response"].content
    msg = cl.Message(content="")    
    # print(response["response"].content)
    # print(f"Number of found context: {len(response['context'])}")
    for i in range(0, len(answer_content), 50):  # Adjust chunk size (e.g., 50 characters)
        chunk = answer_content[i:i+50]
        await msg.stream_token(chunk)

    # Send the response back to the user
    await msg.send()

    context_documents = response["context"]
    num_contexts = len(context_documents)
    context_msg = f"Number of found context: {num_contexts}"
    await cl.Message(content=context_msg).send()