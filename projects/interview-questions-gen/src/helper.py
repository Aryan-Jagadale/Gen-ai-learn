from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from src.prompt import *

llm = Ollama(
        model="llama3.2", 
        base_url="http://127.0.0.1:11434",
        temperature=0.3,
        num_predict=2000,
        top_p=0.9,
        repeat_penalty=1.1
)

def filter_questions(text_list):
    questions = []
    for line in text_list:
        line = line.strip()
        if line.endswith('?'):
            questions.append(line)
    return questions

def file_processing(file_path):
    # file_path = 'data/stats.pdf'
    loader = PyPDFLoader(file_path)
    data = loader.load()
    question_gen = ""
    for page in data:
        question_gen += page.page_content
    
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunk_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content = t) for t in chunk_ques_gen]
    return document_ques_gen


def llm_pipeline(file_path):
    document_ques_gen = file_processing(file_path)
    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])
    REFINE_PROMPT_QUESTIONS = PromptTemplate(input_variables=["existing_answer", "text"],template=refine_template,)
    ques_gen_chain = load_summarize_chain(llm = llm, 
                                          chain_type = "refine", 
                                          verbose = True, 
                                          question_prompt=PROMPT_QUESTIONS, 
                                          refine_prompt=REFINE_PROMPT_QUESTIONS)
    ques = ques_gen_chain.run(document_ques_gen)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    vector_store = FAISS.from_documents(document_ques_gen, embeddings)
    ques_list = ques.split("\n")
    filtered_ques_list = filter_questions(ques_list)
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm, 
                                               chain_type="stuff", 
                                               retriever=vector_store.as_retriever())
    return answer_generation_chain, filtered_ques_list
    
    




    





