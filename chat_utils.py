from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import ChatMode
from doc_utils import load_local_docs, load_github_repo
from model_utils import set_chat_model, set_embedding_model, set_chat_memory


def setup_index_and_chat_engine(docs, embed_model, llm, memory, custom_prompt):
    if len(docs) > 0:
        chat_mode = ChatMode.CONTEXT
    else:
        chat_mode = ChatMode.SIMPLE

    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    chat_prompt = (
        "You are an AI coding assistant, your primary function is to help users with coding-related questions \n"
        "and tasks. You have access to a knowledge base of programming documentation and best practices. \n"
        "When answering questions please follow these guidelines. 1. Provide clear, concise, and \n"
        "accurate code snippets when appropriate. 2. Explain your code and reasoning step by step. 3. Offer \n"
        "suggestions for best practices and potential optimizations. 4. If the user's question is unclear, \n"
        "ask for clarification dont assume or guess the answer to any question. 5. When referencing external \n"
        "libraries or frameworks, briefly explain their purpose. 6. If the question involves multiple possible \n"
        "approaches, outline the pros and cons of each. Always Remember to be friendly! \n"
        "Response:"
    )
    chat_engine = index.as_chat_engine(
        chat_mode=chat_mode,
        memory=memory,
        stream=True,
        system_prompt=chat_prompt if custom_prompt is None else custom_prompt,
        llm=llm,
        verbose=True,
        context_prompt=("Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer \n"
                        "the query in a crisp manner, incase case you don't know the answer say 'I don't know!'. \n"
                        "Query: {query_str} \n"
                        "Answer: ")
    )
    return chat_engine


class ChatEngine:
    def __init__(self):
        self.owner = None
        self.branch = None
        self.repo = None
        self.chat_engine = None

    def create_chat_engine(self):
        embed_model = set_embedding_model()
        llm = set_chat_model()
        docs = load_local_docs()
        if all([self.repo, self.owner, self.branch != ""]):
            docs += load_github_repo(self.owner, self.repo, self.branch)
        memory = set_chat_memory()
        custom_prompt = None
        return setup_index_and_chat_engine(docs, embed_model, llm, memory, custom_prompt)

    def process_input(self, message):
        if self.chat_engine is None:
            self.chat_engine = self.create_chat_engine()
        return self.chat_engine.stream_chat(message)

    def stream_response(self, message, history):
        response = self.process_input(message)
        full_response = ''
        for token in response.response_gen:
            full_response += token
            chat_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_response}
            ]
            yield "", chat_history

    def set_github_info(self, owner, repo, branch):
        self.owner, self.repo, self.branch = owner, repo, branch
        self.reset_chat_engine()

    def reset_github_info(self):
        self.owner = self.repo = self.branch = ""
        self.set_github_info(self.owner, self.repo, self.branch)
        self.reset_chat_engine()
        return self.owner, self.repo, self.branch

    def reset_chat_engine(self):
        self.chat_engine = None

