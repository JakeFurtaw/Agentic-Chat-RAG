# Agentic Chat RAG : Interactive Coding Assistant
This new and improved version of Chat RAG which includes an AI agent to help formulate the best responses 
to your coding questions. This version works with the latest LLMs, the newest PyTorch, and the newest 
version of Gradio. This allows the program to work with the new Nvidia Blackwell GPUs.

## Features

## Setup and Usage
1. Clone the repository.
2. Install the required dependencies.
3. Set up your .env file with the following:
````
GRADIO_TEMP_DIR="YourPathTo/Agentic-Chat-RAG/data"
GRADIO_WATCH_DIRS="YourPathTo/Agentic-Chat-RAG"
HUGGINGFACE_HUB_TOKEN="YOUR HF TOKEN HERE"
GITHUB_PAT="YOUR GITHUB PERSONAL ACCESS TOKEN HERE"
LLAMA_CLOUD_API_KEY="YOUR LLAMA_CLOUD_API_KEY"
TAVILY_API_KEY="YOUR TAVILY API KEY HERE"
````
4. Change the model name of your embedding model to whatever model you want to use or the location of
a downloaded huggingface embedding model. This is located in model_utils.py
```
model_name="..."
```

5. Run the application:
````
gradio acr.py
````
6. The app will automatically open a new tab and launch in your browser.
