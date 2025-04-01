from llama_index.core.agent.react import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex, Document
from model_utils import set_chat_model, set_embedding_model
from doc_utils import load_local_docs, load_github_repo
from bs4 import BeautifulSoup
import os, requests
from tavily import TavilyClient
import dotenv

dotenv.load_dotenv()

class AgentTools:
    def __init__(self):
        self.llm = set_chat_model()
        self.agent = None
        self.tools = []
        self.tavily_client = None
        self.setup_tavily_client()
        self.setup_tools()
        self.create_agent()



    def setup_tavily_client(self):
        """Initialize the Tavily client with API key from environment variables."""
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        else:
            print("Warning: TAVILY_API_KEY not found in environment variables. Tavily search will not be available.")

    def setup_tools(self):
        """Set up all available tools for the agent."""
        # Add tools to the list
        self.tools.append(FunctionTool.from_defaults(fn=self.use_local_files))
        if os.getenv("GITHUB_PAT"):
            self.tools.append(FunctionTool.from_defaults(fn=self.use_github_repo))
        # self.tools.append(FunctionTool.from_defaults(fn=self.use_web_search))
        if self.tavily_client:
            self.tools.append(FunctionTool.from_defaults(fn=self.use_tavily_search))



    def create_agent(self):
        """Create the ReActAgent with the configured tools and LLM."""
        system_prompt = (
            "You are an AI coding assistant with access to specialized tools to help solve complex programming tasks. "
            "Your goal is to understand the user's request and strategically choose the appropriate tools to fulfill it. "
            "\n\n"
            "Available tools:"
            "\n- use_local_files: Search through local project files for relevant code, documentation, or data"
            "\n- use_github_repo: Access and analyze GitHub repositories to find solutions or examples"
        )

        # Add Tavily tool to system prompt if available
        if self.tavily_client:
            system_prompt += "\n- use_tavily_search: Search the web using Tavily's AI search engine to find relevant information"

        system_prompt += (
            "\n\n"
            "Guidelines for using tools effectively:"
            "\n1. ALWAYS consider which tool is most appropriate before responding"
            "\n2. For code questions about the user's files, use the local_files tool first"
            "\n3. For open source projects or examples, use the github_repo tool"
        )

        # Add Tavily guideline if available
        if self.tavily_client:
            system_prompt += "\n5. For up-to-date information or general web searches, use the tavily_search tool"
            # Adjust numbering for subsequent guidelines
            system_prompt += "\n6. You may use multiple tools in sequence to build a comprehensive answer but only if needed"
            system_prompt += "\n7. Think step by step to decompose complex questions into subtasks that can be handled by individual tools"
            system_prompt += "\n8. After gathering information with tools, synthesize it into a clear, concise response"
        else:
            system_prompt += "\n5. You may use multiple tools in sequence to build a comprehensive answer but only if needed"
            system_prompt += "\n6. Think step by step to decompose complex questions into subtasks that can be handled by individual tools"
            system_prompt += "\n7. After gathering information with tools, synthesize it into a clear, concise response"

        system_prompt += (
            "\n\n"
            "When analyzing code, focus on:"
            "\n- Identifying bugs, inefficiencies, or potential improvements"
            "\n- Explaining how the code works and suggesting best practices"
            "\n- Providing complete, working solutions with explanations"
            "\n\n"
            "Always be explicit about your reasoning process and clearly indicate which tools you're using and why."
        )
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=False
        )
        return self.agent

    def run_agent(self, query: str) -> str:
        """Run the agent with the user's query and return the response."""
        if not self.agent:
            self.create_agent()
        return self.agent.stream_chat(query)



    def use_local_files(self, query: str = None, directory: str = "data") -> str:
        """
        Tool to search and extract information from local files.

        Args:
            query (str): The specific information or question to search for in the files.
            directory (str): The directory containing the files to search. Defaults to "data".

        Returns:
            str: Information extracted from local files relevant to the query.
        """
        try:
            if not os.path.exists(directory):
                return f"Directory '{directory}' does not exist."
            documents = load_local_docs()
            if not documents:
                return f"No documents found in directory '{directory}'."
            embed_model = set_embedding_model()
            index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            query_engine = index.as_query_engine()
            if query:
                response = query_engine.query(query)
                return f"Local Files Result: {response.response}"
            else:
                file_count = len(documents)
                file_names = [doc.metadata.get('file_name', 'Unknown') for doc in documents]
                return f"Found {file_count} files: {', '.join(file_names[:5])}{'...' if file_count > 5 else ''}"
        except Exception as e:
            return f"Error accessing local files: {str(e)}"

    def use_github_repo(self, owner: str = None, repo: str = None, branch: str = "main", query: str = None) -> str:
        """
        Tool to search and extract information from a GitHub repository.

        Args:
            owner (str): GitHub repository owner username.
            repo (str): GitHub repository name.
            branch (str): GitHub repository branch name. Defaults to "main".
            query (str): The specific information or question to search for in the repository.

        Returns:
            str: Information extracted from the GitHub repository relevant to the query.
        """
        try:
            if not all([owner, repo]):
                return "GitHub repository owner and name are required."
            documents = load_github_repo(owner, repo, branch)
            if not documents:
                return f"No documents found in GitHub repository {owner}/{repo} (branch: {branch})."
            embed_model = set_embedding_model()
            index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            query_engine = index.as_query_engine()
            if query:
                response = query_engine.query(query)
                return f"GitHub Repo Result: {response.response}"
            else:
                file_count = len(documents)
                file_names = [doc.metadata.get('file_name', 'Unknown') for doc in documents]
                return f"Found {file_count} files in {owner}/{repo}: {', '.join(file_names[:5])}{'...' if file_count > 5 else ''}"

        except Exception as e:
            return f"Error accessing GitHub repository: {str(e)}"


    def use_tavily_search(self, query: str, search_depth: str = "basic", max_results: int = 5) -> str:
        """
        Tool to search the web using Tavily's AI search engine.

        Args:
            query (str): The search query to find information on the web.
            search_depth (str): The depth of the search, either "basic" or "comprehensive".
                               "basic" for faster, simpler searches, "comprehensive" for more thorough results.
            max_results (int): Maximum number of results to return (1-10). Defaults to 5.

        Returns:
            str: Relevant information from the web based on the query.
        """
        try:
            if not self.tavily_client:
                return "Tavily search is not available. Please check if TAVILY_API_KEY is set in the environment variables."

            if not query:
                return "A search query is required for Tavily search."

            # Validate search_depth parameter
            if search_depth not in ["basic", "advanced"]:
                search_depth = "basic"

            # Validate max_results parameter
            max_results = min(max(1, max_results), 10)  # Ensure between 1 and 10

            # Perform the search
            search_result = self.tavily_client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results
            )

            # Extract results
            if not search_result.get("results"):
                return f"No results found for query: {query}"

            # Format the results
            formatted_results = f"Tavily Search Results for: {query}\n\n"

            for i, result in enumerate(search_result.get("results", []), 1):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "No URL")

                formatted_results += f"{i}. {title}\n"
                formatted_results += f"   URL: {url}\n"
                formatted_results += f"   Summary: {content[:200]}...\n\n"

            return formatted_results

        except Exception as e:
            return f"Error using Tavily search: {str(e)}"


    # def use_web_search(self, url: str = None, query: str = None) -> str:
    #     """
    #     Tool to search and extract information from web pages.
    #
    #     Args:
    #         url (str): The URL of the web page to search.
    #         query (str): The specific information or question to search for on the web page.
    #
    #     Returns:
    #         str: Information extracted from the web page relevant to the query.
    #     """
    #     try:
    #         if not url:
    #             return "URL is required for web search."
    #         # Fetch the web page
    #         headers = {
    #             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    #         }
    #         response = requests.get(url, headers=headers)
    #         if response.status_code != 200:
    #             return f"Failed to retrieve content from {url}. Status code: {response.status_code}"
    #         # Parse HTML content
    #         soup = BeautifulSoup(response.text, 'html.parser')
    #         # Remove script and style elements
    #         for script in soup(["script", "style"]):
    #             script.extract()
    #         # Get text
    #         text = soup.get_text()
    #         lines = [line.strip() for line in text.splitlines() if line.strip()]
    #         text = ' '.join(lines)
    #         # Create a Document object
    #         documents = [Document(text=text, metadata={"source": url})]
    #
    #         embed_model = set_embedding_model()
    #         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    #         query_engine = index.as_query_engine()
    #         if query:
    #             response = query_engine.query(query)
    #             return f"Web Search Result: {response.response}"
    #         else:
    #             content_preview = text[:200] + "..." if len(text) > 200 else text
    #             return f"Retrieved content from {url}:\n{content_preview}"
    #     except Exception as e:
    #         return f"Error accessing web page: {str(e)}"


