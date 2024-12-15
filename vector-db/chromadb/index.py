from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA


embeddings = OllamaEmbeddings(model="llama3.2",base_url="http://localhost:11434");

loader = DirectoryLoader("./new_articles/", glob = "./*.txt", loader_cls= TextLoader)
document = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
text = text_splitter.split_documents(document)

# Creating DB
vectorDB = Chroma.from_documents(
    documents=text, 
    embedding=embeddings,
    persist_directory="./ollama_chroma_db"
)

vectorDB = None

vectorDB = Chroma(persist_directory="./ollama_chroma_db",
                  embedding_function=embeddings)
# Make a retiver 
retriever = vectorDB.as_retriever(search_kwargs={"k": 2})
     
docs = retriever.get_relevant_documents("Who was Pando was co-launched by?") # Expected ans: Jayakrishnan and Abhijeet Manohar

llm=Ollama(model="llama3.2",base_url="http://localhost:11434")

print("LLM::>",llm)


qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


#full example
query = "What is the news about Pando?"
llm_response = qa_chain(query)
process_llm_response(llm_response)


vectorDB.delete_collection()
