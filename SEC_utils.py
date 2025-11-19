from langchain.agents import create_agent
from langchain.tools import tool
import random
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import htmltabletomd
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from llm import llm as model

embeddings = NVIDIAEmbeddings(
  model="nvidia/nv-embedcode-7b-v1", 
  api_key="", 
  truncate="NONE", 
)

vector_store = Chroma(
    collection_name="10K_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

os.environ["USER_AGENT"] = "Your Name your_email@example.com"
def scrape_sec_filing(url: str):
    headers = {
    "User-Agent": "Your Name your_email@example.com"   # SEC requires this
    }
    r = requests.get(url, headers=headers)
    # print(r.text)
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "img"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    with open("filing.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return text

def get_SEC_filings(company_name: str, num_filings: int = 5) -> list[dict]:
    """Fetch the latest SEC filings for a given company.
    Args:
        company_name (str): The name of the company to fetch filings for.
    Returns:
        list[dict]: A list of SEC filings.
    """
    cik = get_cik_from_company_name(company_name).zfill(10)
    baseUrl = "https://data.sec.gov/submissions/CIK{cik}.json".format(cik=cik)
    headers = {"User-Agent": "YourName your.email@example.com"}
    filing_types = ["10-K"]
    processed_filings = []
    response = requests.get(baseUrl, headers=headers)
    data = response.json()
    filings = data.get("filings",{}).get("recent",{})
    
    for i in range(len(filings.get("form",[]))):
        if len(processed_filings)>=num_filings:
            break
        if filings.get("form",[])[i] in filing_types:
            processed_filings.append({
                "accession_number": filings["accessionNumber"][i],
                "form_type": filings["form"][i],
                "filing_date": filings["filingDate"][i],
                "report_date": filings.get("reportDate", [""])[i],
                "filing_url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{filings['accessionNumber'][i].replace('-', '')}/{filings.get("primaryDocument", [])[i]}",
                "primary_docs": filings.get("primaryDocument", [])[i],  # may be present
                "primary_doc_descriptions": filings.get("primaryDocDescription", [])[i]
            })
    return processed_filings



def get_cik_from_company_name(company_name: str) -> str:
    """
    Lookup the CIK for a given company name using the SEC's company_tickers.json.
    Args:
        company_name (str): The name of the company.
    Returns:
        str: The CIK as a string, or None if not found.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "YourName your.email@example.com"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
    data = response.json()
    for entry in data.values():
        if entry["title"].lower() == company_name.lower():
            return str(entry["cik_str"])
    return None
def store_10k_content(list_of_filings):

    loader = WebBaseLoader([file['filing_url'] for file in list_of_filings],header_template={"User-Agent": "YourName your.email@example.com"})
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    document_ids = vector_store.add_documents(documents=all_splits)

@tool(response_format='content_and_artifact')
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

from langchain.agents import create_agent


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a 10K document. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)
Í
query = (
    "Which company this 10K filing is about\n\n"
    "What does it say in item 1A?"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values"):
    event["messages"][-1].pretty_print()
