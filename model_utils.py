from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

TOKEN_LIMIT=120000

def set_device(gpu: int = None) -> str:
    return f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu"

def set_chat_model():
    llm = Ollama(
        model='mistral-nemo:latest',
        temperature=.7,
        context_window=120000,
        request_timeout=60,
        keep_alive='30s')
    return llm

def set_embedding_model():
    embed_model = HuggingFaceEmbedding(
        model_name="/home/jake/Programming/Models/embedding/multilingual-e5-large-instruct",
        device=set_device(0), trust_remote_code=True)
    return embed_model

def set_chat_memory():
    token_limit=TOKEN_LIMIT
    return ChatMemoryBuffer.from_defaults(token_limit=token_limit)