# Chat RAG 2.0: Interactive Coding Assistant
This new and improved version of Chat RAG works with the latest LLMs, the newest PyTorch, 
and the newest version of Gradio. This allows the program to work with the new Nvidia Blackwell GPUs 
coming out soon.

## Setup and Usage
1. Clone the repository.
2. Install the required dependencies.
3. Set up your .env file with the following:
````
GRADIO_TEMP_DIR="YourPathTo/Chat-RAG/data"
GRADIO_WATCH_DIRS="YourPathTo/Chat-RAG"
HUGGINGFACE_HUB_TOKEN="YOUR HF TOKEN HERE"
NVIDIA_API_KEY="YOUR NVIDIA API KEY HERE"
OPENAI_API_KEY="YOUR OpenAI API KEY HERE"
ANTHROPIC_API_KEY="YOUR Anthropic API KEY HERE"
GITHUB_PAT="YOUR GITHUB PERSONAL ACCESS TOKEN HERE"
LLAMA_CLOUD_API_KEY="YOUR LLAMA_CLOUD_API_KEY"
````
4. Run the application:
````
gradio chatrag.py
````
5. The app will automatically open a new tab and launch in your browser.
