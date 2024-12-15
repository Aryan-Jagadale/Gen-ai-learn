from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

embeddings = OllamaEmbeddings(model="llama3.2",base_url="http://localhost:11434");

WEAVIATE_API_KEY = ""
WEAVIATE_CLUSTER = ""


loader = DirectoryLoader("./data",glob = "./*.txt", loader_cls= TextLoader)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
text = text_splitter.split_documents(data)

print("data::>",data)