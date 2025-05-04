from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.github import GithubClient, GithubRepositoryReader
import glob, os, dotenv, shutil

dotenv.load_dotenv()

DIRECTORY_PATH = "data"
SUPPORTED_EXTENSIONS = [
    '.pdf', '.602', '.abw', '.cgm', '.cwk', '.doc', '.docx', '.docm', '.dot', '.dotm',
    '.hwp', '.key', '.lwp', '.mw', '.mcw', '.pages', '.pbd', '.ppt', '.pptm', '.pptx',
    '.pot', '.potm', '.potx', '.rtf', '.sda', '.sdd', '.sdp', '.sdw', '.sgl', '.sti',
    '.sxi', '.sxw', '.stw', '.sxg', '.txt', '.uof', '.uop', '.uot', '.vor', '.wpd',
    '.wps', '.xml', '.zabw', '.epub', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
    '.tiff', '.webp', '.htm', '.html', '.xlsx', '.xls', '.xlsm', '.xlsb', '.xlw',
    '.csv', '.dif', '.sylk', '.slk', '.prn', '.numbers', '.et', '.ods', '.fods',
    '.uos1', '.uos2', '.dbf', '.wk1', '.wk2', '.wk3', '.wk4', '.wks', '.123',
    '.wq1', '.wq2', '.wb1', '.wb2', '.wb3', '.qpw', '.xlr', '.eth', '.tsv',
    '.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'
]

def setup_data_directory():
    """Ensure the data directory exists."""
    if not os.path.exists(DIRECTORY_PATH):
        os.makedirs(DIRECTORY_PATH)

def load_local_docs():
    """Load local documents with simplified parser logic."""
    setup_data_directory()
    parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY")) if os.getenv("LLAMA_CLOUD_API_KEY") else None
    all_files = glob.glob(os.path.join(DIRECTORY_PATH, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    documents = []
    for file in all_files:
        try:
            file_extractor = {ext: parser for ext in SUPPORTED_EXTENSIONS} if parser else None
            documents.extend(
                SimpleDirectoryReader(input_files=[file], file_extractor=file_extractor).load_data()
            )
        except Exception as e:
            print(f"Error loading file {file}: {str(e)}")
    return documents

def clear_local_docs():
    """Clear the local data directory."""
    if os.path.exists(DIRECTORY_PATH):
        shutil.rmtree(DIRECTORY_PATH)
    setup_data_directory()

def load_github_repo(owner, repo, branch):
    """Load documents from a GitHub repository."""
    if not os.getenv("GITHUB_PAT"):
        raise ValueError("GitHub Personal Access Token not found in .env file.")
    github_client = GithubClient(github_token=os.getenv("GITHUB_PAT"), verbose=True)
    documents = GithubRepositoryReader(
        github_client=github_client,
        owner=owner,
        repo=repo,
        use_parser=False,
        verbose=False,
        filter_file_extensions=([".png", ".jpg", ".jpeg", ".gif", ".svg"], GithubRepositoryReader.FilterType.EXCLUDE)
    ).load_data(branch=branch)
    return documents