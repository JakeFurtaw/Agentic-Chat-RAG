from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import BitsAndBytesConfig
import torch, gc

TOKEN_LIMIT=120000

def set_device(gpu: int = None) -> str:
    return f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu"

def set_chat_model():

    #TODO This model has reasoning. Use thinking=on or thinking=off and add it to the system prompt in the chat engine to try it
    #this only works for one question need to figure out why i cant have a full conversation...
    torch.cuda.empty_cache()
    gc.collect()
    llm = HuggingFaceLLM(
        model_name="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
        tokenizer_name="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
        context_window=120000,
        max_new_tokens=2500,
        is_chat_model=True,
        device_map=set_device(0),
        generate_kwargs={
            "temperature":0.7,
            "do_sample": True,
        },
        model_kwargs={
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            ),
            "trust_remote_code":True
        }
    )

    # llm = Ollama(
    #     model='llama3.3:latest',
    #     temperature=0.7,
    #     context_window=120000,
    #     request_timeout=60,
    #     keep_alive='30s')# Keeps model alive for 30 seconds after last query
    return llm

def set_embedding_model():
    embed_model = HuggingFaceEmbedding(
        model_name="/home/jake/Programming/Models/embedding/multilingual-e5-large-instruct",  #Change this to reflect where your local embedding model is
        device=set_device(0),
        trust_remote_code=True)
    return embed_model

def set_chat_memory():
    return ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT)