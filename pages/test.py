import datetime
import json
import os
from typing import Dict, List
from collections import defaultdict

import html2text
import openai
import pandas as pd
import plotly.express as px
import PyPDF2
import requests
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from requests_cache import CachedSession
from sec_api import QueryApi

from apikeys import get_secret, ConfigKey
from assurifai.prompts import risk_queries, get_value_queries, financial_keywords
from charts import finapi

ss = st.session_state

# https://github.com/VikParuchuri/marker/issues/442
# i believe is issue with autoreload. i dont think this fixes it
# torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

r = CachedSession("finapi", expire_after=datetime.timedelta(hours=1))

# improvements:
# page based chunking/cleaning/metainfo such as xlbr


openai.api_key = get_secret(ConfigKey.OPENAI)
SEC_API_KEY = get_secret(ConfigKey.SEC_API)


query_api = QueryApi(api_key=SEC_API_KEY)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    # 'Accept-Encoding': 'gzip, deflate, br, zstd',
    "DNT": "1",
    "Sec-GPC": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


@st.cache_data
def fetch_10k_reports(ticker):
    query = {
        "query": {
            "query_string": {
                "query": f'ticker:{ticker} AND filedAt:[2019-01-01 TO 2024-12-31] AND formType:"10-K"'
            }
        },
        "from": "0",
        "size": "6",
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    response = query_api.get_filings(query)

    reports = []
    if response is not None:
        for filing in response.get("filings", []):
            if int(filing["periodOfReport"][:4]) > 2018:
                reports.append(filing)
    return reports


@st.cache_data
def extract_text_from_html(url):
    response = r.get(url, headers=headers)
    if response.status_code == 200:
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        return text_maker.handle(response.text)
    return ""


def fetch_risk_related_info(text_map: Dict[str, List[str]]):
    all_chunks = [chunk for chunks in text_map.values() for chunk in chunks]

    ret_dict = {}

    for key, task in risk_queries.items():
        search, instruction = task
        chunks = [
            chunk
            for chunk in all_chunks
            if any(word in chunk.lower() for word in search)
        ]

        response = """{ "items": [] }"""

        retriever = FAISS.from_texts(chunks, embeddings).as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini"), retriever=retriever
        )
        response = str(qa.run(instruction))
        ret_dict[key] = response[response.index("{") : response.rindex("}") + 1]

    return ret_dict

def fetch_financial_info(text_map: Dict[str, List[str]]):
    st.header("Financial info")

    check = st.button("Run Financial Info")
    if not check:
        return

    text_map = {
        year: [
            chunk
            for chunk in chunks
            if any(word in chunk.lower() for word in financial_keywords)
        ]
        for year, chunks in text_map.items()
    }
    ret_dict = defaultdict(lambda: defaultdict(str))
    with st.expander("See Progress"):
        for year, chunks in list(text_map.items())[:1]:
            st.text(f"{len(chunks)} chunks found")
            retriever = FAISS.from_texts(chunks, embeddings).as_retriever()
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4o-mini"), retriever=retriever
            )

            for key, task in get_value_queries(year).items():
                search, instruction = task
                st.text(key.capitalize())
                response = """{}"""
                with st.spinner("Calling API"):
                    response = str(qa.run(instruction))
                response = response[response.index("{") : response.rindex("}") + 1]
                print(response)
                st.json(response)
                ret_dict[key][year] = json.loads(response)["value"]
    ret_df = pd.DataFrame(ret_dict)
    st.dataframe(ret_df)

def render_main():
    st.set_page_config(layout="wide")
    st.title("📄 PDF Query App")

    ticker = st.text_input("Enter Company Ticker (e.g., AAPL, TSLA):")
    if not ticker:
        return

    graph = finapi.price(ticker).to_frame()
    graph = graph[graph.index >= datetime.date(2019, 1, 1)]
    st.plotly_chart(px.line(graph, title="Stock Price"))

    reports = fetch_10k_reports(ticker)

    if reports:
        st.write("### Last 5 10-K Reports:")
        year_texts = {
            r["periodOfReport"][:4]: extract_text_from_html(r["linkToFilingDetails"])
            for r in reports
        }
        for report in reports:
            st.markdown(
                f"- [{report['periodOfReport'][:4]}]({report['linkToFilingDetails']})"
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        text_map = {}
        for k, text in year_texts.items():
            text_map[k] = [
                f"Report 10-K year: {k} {chunk}"
                for chunk in text_splitter.split_text(text)
            ]
        st.header("Risk related info")

        check = st.button("Run Risk Related Info")
        if check:
            info = None
            with st.spinner("Running..."):
                info = fetch_risk_related_info(text_map)
        fetch_financial_info(text_map)

    else:
        st.write("No reports found.")

        # What is the operating income? Answer as a JSON: (eg.: { "value":  3000000000, reason: "reason here", confidence: 0.98 })


render_main()
