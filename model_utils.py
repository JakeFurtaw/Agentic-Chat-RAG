from llama_index.llms.ollama import Ollama

ollama_llm = Ollama(
    model='mistral-nemo:latest',
    temperature=.7,
    context_window=120000,
    request_timeout=60,
    keep_alive='30s'
)