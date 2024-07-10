import os

#import glob
import json
import logging
import shutil
from pathlib import Path

#import chromadb
#import pandas as pd
#import pypdf
from langchain.chains import RetrievalQAWithSourcesChain  # from retriever
from langchain.chains.qa_with_sources import load_qa_with_sources_chain  # from passed docs
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, PyPDFDirectoryLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
#from langchain.retrievers import ContextualCompressionRetriever
#from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain.vectorstores import Chroma
#from langchain_community.chat_models import ChatOllama
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_openai import OpenAIEmbeddings

# UTILITIES FUNCTIONS


def convert_unix_to_utf8(doc):
    """
    Converts non-breaking space characters (\xa0) to regular spaces in the page content of a document.
    Args:
        doc (list): A list of page objects, where each page object has an attribute `page_content` that is a string.
    Returns:
        list: The modified list of page objects with non-breaking spaces replaced by regular spaces.
    """
    for page in doc:
        page.page_content = page.page_content.replace("\xa0", " ")
    return doc


def get_yes_or_no_input(prompt):
    """
    Continuously prompts the user for a 'yes' or 'no' response and returns the corresponding boolean value.
    Args:
        prompt (str): The prompt message to display to the user.
    Returns:
        bool: True if the user responds with 'yes', False if the user responds with 'no'.
    """
    while True:
        response = input(prompt).strip().lower()
        if response in ('yes', 'no'):
            if response == 'yes':
                return True
            elif response == 'no':
                return False
            else:
                print("Please enter 'yes' or 'no'.")


class Rag():

    """
    A class to manage and retrieve information from a document corpus using language models and embeddings.

    Attributes:
        corpus_path (Path): The path to the document corpus.
        db_path (Path): The path to the vector store database.
        chunk_size (int): The size of chunks for splitting documents. Default is 1024.
        chunk_overlap (int): The overlap between chunks. Default is 256.
        model (str): The name of the language model to use. Default is 'llama2'.
        embedding_function (function): The function to generate embeddings. Default is OpenAIEmbeddings().
        score_thresh (float): The score threshold for similarity search. Default is 0.55.
        llm_temperature (int): The temperature for the language model. Default is 0.
        k (int): The number of top results to retrieve. Default is 3.

    Methods:
        load_and_split_corpus():
            Loads the document corpus and splits it into chunks.
        
        build_vector_store():
            Builds or updates the vector store database with document embeddings.
        
        retrieve_from_docs(query):
            Retrieves information from the document corpus based on a query.
        
        retrieve_from_retriever(query):
            Retrieves information using a retriever chain based on a query.
    """

    def __init__(self,
                 corpus_path,
                 db_path,
                 chunk_size=2024,
                 chunk_overlap=256,
                 model='llama2',
                 embedding_function=OpenAIEmbeddings(),
                 score_thresh=0.55,
                 llm_temperature=0,
                 k=3):
        
        self.corpus_path: Path = corpus_path
        self.db_path: Path = db_path
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.model: str = model #model name, e.g. llama2
        self.llm = Ollama(model=self.model, temperature=llm_temperature)
        self.embedding_function: embedding_function = embedding_function
        self.docs: list = []
        self.search_kwargs: dict = {"score_threshold":score_thresh, "k":k}
        self.result: dict = {}
        self._response: bool = None
        self.template = """Given the following sections from various documents and a question, 
        generate a final answer with references ("SOURCES"). If the answer is unknown, 
        indicate as such without attempting to fabricate a response. Ensure to always 
        include a "SOURCES" section in your answer.

        QUESTION: {question}
        =========
        {summaries}
        =========
        FINAL ANSWER:"""

    def load_all_docs(self):

        if os.path.isdir(self.corpus_path):
            print('[INFO] The provided corpus_path is a directory.')
            print('[INFO] Loading docs..')
            self.__docs = PyPDFDirectoryLoader(self.corpus_path).load()
            print('[INFO] Loaded all directory files.')
            #return self._docs
        
        elif os.path.isfile(self.corpus_path):
            print('[INFO] The provided corpus_path is a file')
            print('[INFO] Loading docs..')
            self.__docs = PyPDFLoader(self.corpus_path).load()
            print('[INFO] Loaded file.')
            #return self._docs
        
        else:
            raise Exception("[Alert] Failed to load the corpus!")

        
    def split_corpus(self):

        print(f"[INFO] Splitting docs..")

        #print(f"[INFO] loaded doc has {len(_docs)} pages")

        # Convert docs from Unix to UTF-8
        self.__docs = convert_unix_to_utf8(self.__docs)

        # Splitting docs into doc chunks
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True)

        self.docs = _text_splitter.split_documents(self.__docs)
        print(f"[INFO] Docs split.")


    def build_vector_store(self):
        # Remove old DB, if requested, create a new db
        if os.path.exists(self.db_path):
            print(f"[INFO] Clearing existing CHROMA DB? ('yes' or 'no')\n  > {self.db_path}")
            self._response = get_yes_or_no_input("Do you want to clear existing DB? (yes/no): ")

            if self._response:
                print('[INFO] Existing DB will be deleted.')
                shutil.rmtree(self.db_path)
                print('[INFO] DB deleted.')

            else:
                print('[INFO] Existing DB will be preserved.')

        else:
            print("[INFO] No DB found") 
            # a new one will be created for both this option and input 'yes' option
            # set to True, in order to initiat the creation of a new DB
            self._response = True

              
        if self._response:
            print('[INFO] New Vector Store DB will be created..')
            print('[INFO] Generating embeddings..')
            self.db = Chroma.from_documents(
                self.docs,
                self.embedding_function,
                persist_directory=self.db_path)

            self.db.persist()
            print('[INFO] Vector Store created!')

    def retrieve_from_vector_db(self, query, score_threshold, top_k):

        _db_conn = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function)
           
        _retriever_db_conn = _db_conn.as_retriever()

        if (score_threshold is None) & (top_k is None):
            _search_kwargs = self.search_kwargs()

        elif (score_threshold is None) & (top_k is not None):
            _search_kwargs = self.search_kwargs()
            _search_kwargs['score_threshold'] = score_threshold
        
        elif (score_threshold is not None) & (top_k is None):
            _search_kwargs = self.search_kwargs()
            _search_kwargs['k'] = top_k
        
        else:
            _search_kwargs = {'score_threshold':score_threshold, 'k':top_k}

        print(f"[INFO] Retrieving with search_kwargs:\n{json.dumps(_search_kwargs, indent=4)}")

        self.result = _retriever_db_conn.get_relevant_documents(query, search_kwargs=_search_kwargs)

        for _relevant_doc in self.result:
            print("-"*72)
            print(_relevant_doc.metadata)
            print("\n")
            print(_relevant_doc.page_content)

    def retrieve_from_docs(self, query: str):
        
        _prompt1 = ChatPromptTemplate.from_template(self.template)
        _chain_step_1 = load_qa_with_sources_chain(
            llm=self.llm,
            prompt=_prompt1,
            output_key="summary")
        
        _db_conn = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function)
                
        self.docs = _db_conn.similarity_search(
            query, 
            search_type="similarity_score_threshold", 
            search_kwargs=self.search_kwargs)
        
        _chain_step_1.run(
            input_documents=self.docs, 
            question=query)
        

    def retrieve_from_retriever(self, query):

        _db_conn = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function)

        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["summaries", "question"],)
        
        _chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt},
            retriever=_db_conn.as_retriever(search_type="similarity_score_threshold", search_kwargs=self.search_kwargs),
            return_source_documents=True,
            verbose=True)
        
        self.result = _chain.invoke(query, return_only_outputs=True)

        print(self.result['answer'])

        for _relevant_doc in self.result['source_documents']:
            print("-"*72)
            print(_relevant_doc.metadata)
            print("\n")
            print(_relevant_doc.page_content)
        
        

def main():
    """
    example use case
    """

    # API Key

    with open("../scr/secrets.txt", 'r', encoding="utf-8") as f:
        api_key = f.readlines()[0]
    os.environ['OPENAI_API_KEY'] = api_key

    CHROMA_PATH = "../db_test_2024_256/"
    #fpath_file = Path('..',' HB','ECSS-E-HB-11A(1March2017).pdf')
    fpath_file = Path('..',' HB','*.pdf')
    test_rag = Rag(db_path=CHROMA_PATH, corpus_path=fpath_file, model='mistral')
    test_rag.retrieve_from_retriever(query="What is resonance search?")


if __name__ == "__main__":

    main()
