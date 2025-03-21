### This app is in production and doesn't have the same features as version one yet! Give me some time
# Agentic Chat RAG : Interactive Coding Assistant
This new and improved version of Chat RAG which includes agents to help formulate the best responses to your
coding questions. This version works with the latest LLMs, the newest PyTorch, and the newest version of 
Gradio. This allows the program to work with the new Nvidia Blackwell GPUs coming out soon.

## Setup and Usage
1. Clone the repository.
2. Install the required dependencies.
3. Set up your .env file with the following:
````
GRADIO_TEMP_DIR="YourPathTo/ChatRAG2.0/data"
GRADIO_WATCH_DIRS="YourPathTo/ChatRAG2.0"
HUGGINGFACE_HUB_TOKEN="YOUR HF TOKEN HERE"
GITHUB_PAT="YOUR GITHUB PERSONAL ACCESS TOKEN HERE"
LLAMA_CLOUD_API_KEY="YOUR LLAMA_CLOUD_API_KEY"
````
4. Change the model name of your embedding model to whatever model you want to use or the location of
a downloaded huggingface model. This is located in model_utils.py
```
model_name="..."
```

5. Run the application:
````
gradio chatrag.py
````
6. The app will automatically open a new tab and launch in your browser.
