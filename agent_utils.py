from llama_index.core.agent.react import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from model_utils import set_chat_model, set_embedding_model, set_chat_memory
from doc_utils import load_local_docs, load_github_repo
from tavily import TavilyClient
import os, dotenv


dotenv.load_dotenv()

class AgentTools:
    def __init__(self):
        self.llm = set_chat_model()
        self.embed_model = set_embedding_model()
        self.chat_memory = set_chat_memory()
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None
        self.local_index = None
        self.github_index = None
        self.agent = None
        self.create_agent()

    def create_agent(self):
        """Create the ReActAgent with tools and a concise system prompt."""
        tools = [
            FunctionTool.from_defaults(fn=self.use_local_files, name='Local_File_Tool'),
        ]
        if os.getenv("GITHUB_PAT"):
            tools.append(FunctionTool.from_defaults(fn=self.use_github_repo, name='GitHub_Repo_Tool'))
        if self.tavily_client:
            tools.append(FunctionTool.from_defaults(fn=self.use_tavily_search))

        system_prompt = (
            "You are an AI coding assistant with access to specialized tools to help solve complex programming tasks. "
            "Your goal is to understand the user's query and provide a clear and concise answer. Only use tools if you need to"
            "If you need to use tools strategically choose the appropriate tools to answer the query."
            "\n\n"
            "Tools:\n"
            "- use_local_files: If query asks about files or to Search local files for code/data use this tool\n"
            "- use_github_repo: If the query asks to Analyze GitHub repos use this tool\n"
            f"{'- use_tavily_search: Web search via Tavily only if both other tools dont help or if '
               'query refers to something that knowledge base doesnt cover\n' if self.tavily_client else ''}"
            "Guidelines:\n"
            "1. If you need to use tools which you will not always need to!!"
            "2. Choose the best tool for the query based off of what the query says\n"
            "3. Use local_files for file-related questions. always use this tool first\n"
            "4. Use github_repo for repo-related queries\n"
            f"{'5. Use tavily_search for web info if needed and only as a last result.\n' if self.tavily_client else ''}"
            "6. Explain code, suggest improvements, and provide working solutions\n"
            "7. Be explicit about your reasoning and tool usage"
        )

        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            memory=self.chat_memory,
            system_prompt=system_prompt,
            verbose=False,
            max_iterations=5
        )
        return self.agent

    def run_agent(self, query: str):
        response = self.agent.stream_chat(query)
        return response

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
        try:
            documents = load_local_docs()
            return self._query_documents(documents, query)
        except Exception as e:
            return f"Error accessing local files: {str(e)}"

    def use_github_repo(self, owner: str = None, repo: str = None, branch: str = None, query: str = None) -> str:
        """Search a GitHub repo for relevant info."""
        try:
            if not all([owner, repo, branch]):
                return "GitHub repository owner branch and name are required."
            documents = load_github_repo(owner, repo, branch)
            return self._query_documents(documents, query)
        except Exception as e:
            return f"Error accessing GitHub repository: {str(e)}"

    def use_tavily_search(self, query: str, search_depth: str = "basic", max_results: int = 5) -> str:
        """Search the web using Tavily."""
        try:
            if not query:
                return "A search query is required."
            if search_depth not in ["basic", "advanced"]:  #might want to add a way to change between basic and advanced
                search_depth = "basic"
            max_results = min(max(1, max_results), 10)
            search_result = self.tavily_client.search(query=query,
                                                      search_depth=search_depth,
                                                      max_results=max_results)
            if not search_result.get("results"):
                return f"No results found for query: {query}"
            formatted_results = f"Tavily Results for: {query}\n\n"
            for i, result in enumerate(search_result["results"], 1):
                formatted_results += f"{i}. {result.get('title', 'No title')}\n   URL: {result.get('url', 'No URL')}\n   Summary: {result.get('content', 'No content')[:200]}...\n\n"
            return formatted_results
        except Exception as e:
            return f"Error using Tavily search: {str(e)}"














#Old Prompt
        # system_prompt = (
        #     "You are an AI coding assistant with access to specialized tools to help solve complex programming tasks. "
        #     "Your goal is to understand the user's query and provide a clear and concise answer. Only use tools if you need to"
        #     "If you dont need to use tools to answer a query than dont use them use your knowledge."
        #     "If you need to use tools strategically choose the appropriate tools to answer the query."
        #     "\n\n"
        #     "Available tools:"
        #     "\n- use_local_files: Search through local project files for relevant code, documentation, or data"
        #     "\n- use_github_repo: Access and analyze GitHub repositories to find solutions or examples"
        # )
        #
        # # Add Tavily tool to system prompt if available
        # if self.tavily_client:
        #     system_prompt += "\n- use_tavily_search: Search the web using Tavily's AI search engine to find relevant information"
        #
        # system_prompt += (
        #     "\n\n"
        #     "Guidelines for using tools effectively:"
        #     "\n1. ALWAYS consider which tool is most appropriate before responding"
        #     "\n2. For code questions about the user's files or if the user mentions a file, use the local_files tool first"
        #     "\n3. For open source projects or examples or if a user mentions a repo or repository, use the github_repo tool"
        # )
        #
        # # Add Tavily guideline if available
        # if self.tavily_client:
        #     system_prompt += "\n5. For up-to-date information or general web searches, use the tavily_search tool. Use this only if you really need to."
        #     # Adjust numbering for subsequent guidelines
        #     system_prompt += "\n6. You may use multiple tools in sequence to build a comprehensive answer but only if needed"
        #     system_prompt += "\n7. Think step by step to decompose complex questions into subtasks that can be handled by individual tools"
        #     system_prompt += "\n8. After gathering information with tools, synthesize it into a clear, concise response"
        # else:
        #     system_prompt += "\n5. You may use multiple tools in sequence to build a comprehensive answer but only if needed"
        #     system_prompt += "\n6. Think step by step to decompose complex questions into subtasks that can be handled by individual tools"
        #     system_prompt += "\n7. After gathering information with tools, synthesize it into a clear, concise response"
        #
        # system_prompt += (
        #     "\n\n"
        #     "When analyzing code, focus on:"
        #     "\n- Identifying bugs, inefficiencies, or potential improvements"
        #     "\n- Explaining how the code works and suggesting best practices"
        #     "\n- Providing complete, working solutions with explanations"
        #     "\n\n"
        #     "Always be explicit about your reasoning process and clearly indicate which tools you're using and why."
        # )