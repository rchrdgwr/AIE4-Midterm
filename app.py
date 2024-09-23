import chainlit as cl
import os
from classes.app_state import AppState
from classes.model_run_state import ModelRunState
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from operator import itemgetter
from utilities.doc_utilities import get_documents
from utilities.templates import get_qa_prompt
from utilities.vector_utilities import create_vector_store

document_urls = [
    "https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf",
     "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
]

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup our state and read the documents
app_state = AppState()
app_state.set_debug(False)
app_state.set_document_urls(document_urls)

get_documents(app_state)

# set up this model run
chainlit_state = ModelRunState()
chainlit_state.name = "Chainlit"

chainlit_state.qa_model_name = "gpt-4o-mini" 
chainlit_state.qa_model = ChatOpenAI(model=chainlit_state.qa_model_name, openai_api_key=openai_api_key)

hf_username = "rchrdgwr"
hf_repo_name = "finetuned-arctic-model"

chainlit_state.embedding_model_name = f"{hf_username}/{hf_repo_name}"
chainlit_state.embedding_model = HuggingFaceEmbeddings(model_name=chainlit_state.embedding_model_name)

chainlit_state.chunk_size = 1000
chainlit_state.chunk_overlap = 100
create_vector_store(app_state, chainlit_state )

chat_prompt = get_qa_prompt()

# create the chain

retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | chainlit_state.retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))


    | {"response": chat_prompt | chainlit_state.qa_model, "context": itemgetter("context")}
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
    MAX_PREVIEW_LENGTH = 100

    response = retrieval_augmented_qa_chain.invoke({"question" : message.content })
    answer_content = response["response"].content
    msg = cl.Message(content="")    

    for i in range(0, len(answer_content), 50):  # Adjust chunk size (e.g., 50 characters)
        chunk = answer_content[i:i+50]
        await msg.stream_token(chunk)

    # Send the response back to the user
    await msg.send()

    context_documents = response["context"]
    num_contexts = len(context_documents)
    context_msg = f"Number of found context: {num_contexts}"


    await cl.Message(content=context_msg).send()

    for doc in context_documents:
        document_title = doc.metadata.get("source", "Unknown Document") 
        chunk_number = doc.metadata.get("chunk_number", "Unknown Chunk")  

        document_context = doc.page_content.strip() 
        truncated_context = document_context[:MAX_PREVIEW_LENGTH] + ("..." if len(document_context) > MAX_PREVIEW_LENGTH else "")
        print("----------------------------------------")
        print(truncated_context)

        await cl.Message(
            content=f"**{document_title} ( Chunk: {chunk_number})**",
            elements=[
                cl.Text(content=truncated_context, display="inline")  
            ]
        ).send()