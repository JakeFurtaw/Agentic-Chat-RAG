# Agentic Chat RAG : Interactive Coding Assistant
This new and improved version of Chat RAG which includes an AI agent to help formulate the best responses 
to your coding questions. This version works with the latest LLMs, the newest PyTorch, and the newest 
version of Gradio. This allows the program to work with the new Nvidia Blackwell GPUs.

## Features
### Chat Mode
- **Chat Mode:** Standard mode where the model answers your queries using:
  - Its built-in knowledge
  - Your local files (if uploaded)
  - A GitHub repository (if linked)

- **Key Features:**
  - **Chat With Files:** Upload documents to provide additional context
  - **Chat with GitHub Repo:** Use files from a GitHub repository as context
  - **Advanced File Support:** Parse .pdf, .csv, .xlsx, .docx, and .xml files using Llama Parse
  - **Interactive Interface:** User-friendly chat experience for all queries
  - **RAG-powered Responses:** Retrieve and generate answers using your uploaded documents or GitHub repository


### Agent Mode
**Agent Mode:** Agent mode empowers the model to utilize three tools to provide the best possible answer to your query.  

***The model follows a sequential approach:***
  1. First, it searches your local documents (if provided)
  2. If insufficient information is found, it checks your GitHub repository (if provided)
  3. Finally, if needed, it searches the internet to gather relevant information

This multi-tool approach ensures comprehensive responses by leveraging all available resources.

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
gradio chatrag.py
````
6. The app will automatically open a new tab and launch in your browser.
