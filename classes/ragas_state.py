import pprint
from ragas.testset.evolutions import simple, reasoning, multi_context
class RagasState:
    def __init__(self):
        self.chunk_size = 600
        self.chunk_overlap = 50
        self.chunks = []
        self.generator_llm = "gpt-4"
        self.critic_llm = "gpt-4o-mini"
        self.distributions = {
            simple: 0.5,
            multi_context: 0.4,
            reasoning: 0.1
        }
        self.num_questions = 3
        self.testset_df = None

