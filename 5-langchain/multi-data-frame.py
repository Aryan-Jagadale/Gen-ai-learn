from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama 
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os
import pandas as pd
import requests
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


llm=Ollama(model="llama3.2",base_url="http://localhost:11434")

url = "https://raw.githubusercontent.com/adamerose/datasets/master/titanic.csv"
df=pd.read_csv(url)
agent=create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
agent.run("How many rows are there")

