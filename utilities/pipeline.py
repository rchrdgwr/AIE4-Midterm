from utilities_2.vectordatabase import VectorDatabase

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm, vector_db_retriever: VectorDatabase, 
                 system_role_prompt, user_role_prompt
        ) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.system_role_prompt = system_role_prompt
        self.user_role_prompt = user_role_prompt

    async def arun_pipeline(self, user_query: str):
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)

        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = self.system_role_prompt.create_message()

        formatted_user_prompt = self.user_role_prompt.create_message(question=user_query, context=context_prompt)

        async def generate_response():
            async for chunk in self.llm.astream([formatted_system_prompt, formatted_user_prompt]):
                yield chunk

        return {"response": generate_response(), "context": context_list}