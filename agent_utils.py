import logging
import os
import dotenv
from llama_index.core.agent.react import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from model_utils import set_chat_model, set_embedding_model, set_chat_memory
from doc_utils import load_local_docs, load_github_repo
from tavily import TavilyClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

class AgentTools:
    def __init__(self, owner=None, repo=None, branch=None):
        self.llm = set_chat_model()
        self.embed_model = set_embedding_model()
        self.chat_memory = set_chat_memory()
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.documents = self.load_documents()
        self.agent = None
        self.create_agent()

    def load_documents(self):
        """Load local and GitHub documents based on configuration."""
        documents = load_local_docs()
        if all([self.owner, self.repo, self.branch]):
            try:
                documents += load_github_repo(self.owner, self.repo, self.branch)
            except Exception as e:
                logger.error("Failed to load GitHub documents: %s", str(e))
        return documents

    def create_agent(self):
        """Create the ReActAgent with tools and a concise system prompt."""
        tools = [
            FunctionTool.from_defaults(fn=self.use_local_files, name='Local_File_Tool'),
            FunctionTool.from_defaults(fn=self.generate_code, name='Code_Generation_Tool'),
        ]
        if os.getenv("GITHUB_PAT"):
            tools.append(FunctionTool.from_defaults(fn=self.use_github_repo, name='GitHub_Repo_Tool'))
        if self.tavily_client:
            tools.append(FunctionTool.from_defaults(fn=self.use_tavily_search))

        system_prompt = (
            "You are an AI coding assistant designed to solve programming tasks. Answer queries clearly and concisely, using tools only when necessary. "
            "Choose the most appropriate tool based on the query content:\n"
            "- Local_File_Tool: For queries about local files or code in the project.\n"
            "- GitHub_Repo_Tool: For queries about GitHub repositories.\n"
            f"{'- use_tavily_search: For web searches when local or GitHub tools are insufficient.\n' if self.tavily_client else ''}"
            "- Code_Generation_Tool: For generating new code snippets.\n"
            "Guidelines:\n"
            "1. Default to LLM knowledge if tools are not needed.\n"
            "2. Explain reasoning and tool usage explicitly.\n"
            "3. Provide complete, working code solutions with explanations.\n"
            "4. Suggest best practices and optimizations."
        )

        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            memory=self.chat_memory,
            system_prompt=system_prompt,
            verbose=True,
            max_iterations=5
        )
        logger.info("Agent created with tools: %s", [tool.metadata.name for tool in tools])
        return self.agent

    def run_agent(self, query: str):
        logger.info("Running agent with query: %s", query)
        try:
            response = self.agent.stream_chat(query)
            return response
        except Exception as e:
            logger.error("Agent query failed: %s", str(e))
            return f"Error processing query: {str(e)}"

    def _query_documents(self, documents, query=None):
        """Helper to query a document set and return results."""
        if not documents:
            return "No documents found."
        index = VectorStoreIndex.from_documents(documents, embed_model=self.embed_model)
        query_engine = index.as_query_engine(llm=self.llm)
        if query:
            response = query_engine.query(query)
            return f"Result: {response}"
        file_count = len(documents)
        file_names = [doc.metadata.get('file_name', 'Unknown') for doc in documents]
        return f"Found {file_count} files: {', '.join(file_names[:5])}{'...' if file_count > 5 else ''}"

    def use_local_files(self, query: str = None, directory: str = "data") -> str:
        """Search local files for relevant info."""
        logger.info("Using local files tool with query: %s", query)
        try:
            return self._query_documents(self.documents, query)
        except Exception as e:
            logger.error("Local files tool error: %s", str(e))
            return f"Error accessing local files: {str(e)}"

    def use_github_repo(self, owner: str = None, repo: str = None, branch: str = None, query: str = None) -> str:
        """Search a GitHub repo for relevant info."""
        logger.info("Using GitHub repo tool with owner: %s, repo: %s, branch: %s, query: %s", owner, repo, branch, query)
        try:
            if not all([owner, repo, branch]):
                return "GitHub repository owner, branch, and name are required."
            documents = load_github_repo(owner, repo, branch)
            return self._query_documents(documents, query)
        except Exception as e:
            logger.error("GitHub repo tool error: %s", str(e))
            return f"Error accessing GitHub repository: {str(e)}"

    def use_tavily_search(self, query: str, max_results: int = 5) -> str:
        """Search the web using Tavily."""
        logger.info("Using Tavily search with query: %s", query)
        try:
            if not query:
                return "A search query is required."
            max_results = min(max(1, max_results), 10)
            search_result = self.tavily_client.search(query=query, search_depth="basic", max_results=max_results)
            if not search_result.get("results"):
                return f"No results found for query: {query}"
            formatted_results = f"Tavily Results for: {query}\n\n"
            for i, result in enumerate(search_result["results"], 1):
                formatted_results += f"{i}. {result.get('title', 'No title')}\n   URL: {result.get('url', 'No URL')}\n   Summary: {result.get('content', 'No content')[:200]}...\n\n"
            return formatted_results
        except Exception as e:
            logger.error("Tavily search error: %s", str(e))
            return f"Error using Tavily search: {str(e)}"

    def generate_code(self, query: str) -> str:
        """Generate code based on the user's query."""
        logger.info("Generating code for query: %s", query)
        try:
            prompt = f"Generate a complete, working code solution for the following request: {query}. Include comments explaining the code and suggest best practices."
            response = self.llm.complete(prompt)
            return f"Generated Code:\n\n{response.text}"
        except Exception as e:
            logger.error("Code generation error: %s", str(e))
            return f"Error generating code: {str(e)}"