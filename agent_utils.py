from llama_index.core.agent.react import ReActAgent
from llama_index.core.tools import FunctionTool
# from llama_index.readers.web import SimpleWebPageReader
from model_utils import set_chat_model
from doc_utils import load_local_docs, load_github_repo
import os


class AgentTools:
    def __init__(self):
        self.llm = set_chat_model()
        self.agent = None
        self.tools = []
        self.setup_tools()
        self.create_agent()

    def setup_tools(self):
        """Set up all available tools for the agent."""
        # Add tools to the list
        self.tools.append(FunctionTool.from_defaults(fn=self.use_local_files))
        self.tools.append(FunctionTool.from_defaults(fn=self.use_github_repo))
        self.tools.append(FunctionTool.from_defaults(fn=self.use_web_search))

    def create_agent(self):
        """Create the ReActAgent with the configured tools and LLM."""
        system_prompt = (
            "You are an AI coding assistant with access to specialized tools to help solve complex programming tasks. "
            "Your goal is to understand the user's request and strategically choose the appropriate tools to fulfill it. "
            "\n\n"
            "Available tools:"
            "\n- use_local_files: Search through local project files for relevant code, documentation, or data"
            "\n- use_github_repo: Access and analyze GitHub repositories to find solutions or examples"
            "\n- use_web_search: Look up information from specific web pages to answer technical questions"
            "\n\n"
            "Guidelines for using tools effectively:"
            "\n1. ALWAYS consider which tool is most appropriate before responding"
            "\n2. For code questions about the user's files, use the local_files tool first"
            "\n3. For open source projects or examples, use the github_repo tool"
            "\n4. For documentation or technical articles, use the web_search tool"
            "\n5. You may use multiple tools in sequence to build a comprehensive answer"
            "\n6. Think step by step to decompose complex questions into subtasks that can be handled by individual tools"
            "\n7. After gathering information with tools, synthesize it into a clear, concise response"
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
            verbose=True
        )

        return self.agent

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

            from llama_index.core import VectorStoreIndex
            from model_utils import set_embedding_model

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

            from llama_index.core import VectorStoreIndex
            from model_utils import set_embedding_model

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
    #
    #         # documents = SimpleWebPageReader().load_data([url])
    #         if not documents:
    #             return f"No content found at URL: {url}"
    #
    #         from llama_index.core import VectorStoreIndex
    #         from model_utils import set_embedding_model
    #
    #         embed_model = set_embedding_model()
    #         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    #         query_engine = index.as_query_engine()
    #
    #         if query:
    #             response = query_engine.query(query)
    #             return f"Web Search Result: {response.response}"
    #         else:
    #             content_preview = documents[0].text[:200] + "..." if len(documents[0].text) > 200 else documents[0].text
    #             return f"Retrieved content from {url}:\n{content_preview}"
    #
    #     except Exception as e:
    #         return f"Error accessing web page: {str(e)}"

    def run_agent(self, query: str) -> str:
        """Run the agent with the user's query and return the response."""
        if not self.agent:
            self.create_agent()

        response = self.agent.chat(query)
        return response

