from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.github import GithubClient, GithubRepositoryReader
import glob, os, dotenv, torch

dotenv.load_dotenv()

DIRECTORY_PATH = "data"

# Local Document Loading Function
def load_local_docs():
    parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))
    all_files = glob.glob(os.path.join(DIRECTORY_PATH, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    documents = []
    supported_extensions = [".pdf", ".docx", ".xlsx", ".csv", ".xml", ".html", ".json"]
    for file in all_files:
        file_extension = os.path.splitext(file)[1].lower()
        if "LLAMA_CLOUD_API_KEY" in os.environ and file_extension in supported_extensions:
            file_extractor = {file_extension: parser}
            documents.extend(
                SimpleDirectoryReader(input_files=[file], file_extractor=file_extractor).load_data())
        else:
            documents.extend(SimpleDirectoryReader(input_files=[file]).load_data())
    return documents

# GitHub Repo Reader setup function. Sets all initial parameters and handles data load of the repository
def load_github_repo(owner, repo, branch):
    if "GITHUB_PAT" in os.environ:
        github_client = GithubClient(github_token=os.getenv("GITHUB_PAT"), verbose=True)
        owner=owner
        repo=repo
        branch=branch
        documents= GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            use_parser=False,
            verbose=False,
            filter_file_extensions=([".png", ".jpg", ".jpeg", ".gif", ".svg"],
                                    GithubRepositoryReader.FilterType.EXCLUDE)
        ).load_data(branch=branch)
        return documents
    else:
        print("Couldn't find your GitHub Personal Access Token in the environment file. Make sure you enter your "
              "GitHub Personal Access Token in the .env file.")