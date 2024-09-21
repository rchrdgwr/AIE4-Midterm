import os
from chainlit.types import AskFileResponse

from utilities_2.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)
from utilities_2.openai_utils.embedding import EmbeddingModel
from utilities_2.vectordatabase import VectorDatabase
from utilities_2.openai_utils.chatmodel import ChatOpenAI
import chainlit as cl
from utilities.text_utils import FileLoader
from utilities.pipeline import RetrievalAugmentedQAPipeline
# from utilities.vector_database import QdrantDatabase


def process_file(file, use_rct): 
    fileLoader = FileLoader()
    return fileLoader.load_file(file, use_rct)

system_template = """\
Use the following context to answer a users question. 
If you cannot find the answer in the context, say you don't know the answer. 
The context contains the text from a document. Refer to it as the document not the context.
"""
system_role_prompt = SystemRolePrompt(system_template)

user_prompt_template = """\
Context:
{context}

Question:
{question}
"""
user_role_prompt = UserRolePrompt(user_prompt_template)

@cl.on_chat_start
async def on_chat_start():
    # get user inputs
    res = await cl.AskActionMessage(
        content="Do you want to use Qdrant?",
        actions=[
            cl.Action(name="yes", value="yes", label="✅ Yes"),
            cl.Action(name="no", value="no", label="❌ No"),
        ],
    ).send()
    use_qdrant = False
    use_qdrant_type = "Local"
    if res and res.get("value") == "yes":
        use_qdrant = True
        local_res = await cl.AskActionMessage(
                content="Do you want to use local or cloud?",
                actions=[
                    cl.Action(name="Local", value="Local", label="✅ Local"),
                    cl.Action(name="Cloud", value="Cloud", label="❌ Cloud"),
                ],
            ).send()
        if local_res and local_res.get("value") == "Cloud":
            use_qdrant_type = "Cloud"
    use_rct = False
    res = await cl.AskActionMessage(
        content="Do you want to use RecursiveCharacterTextSplitter?",
        actions=[
            cl.Action(name="yes", value="yes", label="✅ Yes"),
            cl.Action(name="no", value="no", label="❌ No"),
        ],
    ).send()
    if res and res.get("value") == "yes":
        use_rct = True
    
    files = None
    # Wait for the user to upload a file
    while not files:
        files = await cl.AskFileMessage(
            content="Please upload a .txt or .pdf file to begin processing!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=2,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    texts = process_file(file, use_rct)

    msg = cl.Message(
        content=f"Resulted in {len(texts)} chunks", disable_human_feedback=True
    )
    await msg.send()

    # decide if to use the dict vector store of the Qdrant vector store
    
    # Create a dict vector store
    if use_qdrant == False:
        vector_db = VectorDatabase()
        vector_db = await vector_db.abuild_from_list(texts)
    else:
        embedding_model = EmbeddingModel(embeddings_model_name= "text-embedding-3-small", dimensions=1000)
        if use_qdrant_type == "Local":
            from utilities.vector_database import QdrantDatabase
            vector_db = QdrantDatabase(
                embedding_model=embedding_model 
            )

            vector_db = await vector_db.abuild_from_list(texts)
        
    msg = cl.Message(
        content=f"The Vector store has been created", disable_human_feedback=True
    )
    await msg.send()

    chat_openai = ChatOpenAI()

    # Create a chain
    retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai,
        system_role_prompt=system_role_prompt,
        user_role_prompt=user_role_prompt
    )
    
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` is complete."
    await msg.update()
    msg.content = f"You can now ask questions about `{file.name}`."
    await msg.update()
    cl.user_session.set("chain", retrieval_augmented_qa_pipeline)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    result = await chain.arun_pipeline(message.content)

    async for stream_resp in result["response"]:
        await msg.stream_token(stream_resp)

    await msg.send()