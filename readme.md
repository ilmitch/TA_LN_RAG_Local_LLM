## ZHAW CAS-MAIN Spring 2024 Text Analytics 

# Space Equipment Development Standards: RAG, based on local LLMs (Ollama)

## Overview

This repository contains code and resources for the **TA_LN_RAG_Local_LLM** project, developed as part of the ZHAW CAS-MAIN Text Analytics module. The project focuses on utilizing Retrieval-Augmented Generation (RAG) with local Language Models (LLMs) to enhance information retrieval from the ECSS standards for space equipment development.

![Cover Page](pptx/Screenshot_2024-06-23.jpg)

For more details, please check the [full PDF document](pptx/CAS_MAIN_TA_LN_ECSS-RAG_forDelivery.pdf)

## Contents

- **py_rag**: Python library, abstraction layer on top of Langchain and ChromaDB, to loading, splitting, creating embeddings and querying.
- **./embeddings/**: Contains the embeddings of to ECSS standards, produced with OpenAIEmbeddings(text-embedding-ada-002).
- **01a_ecss_st_rag_v0.1.ipynb**: Jupyter notebook with example usage of the py_rag.Rag() class.
- **01b_eval_v1.0.ipynb**: Jupyter used for hyperparam study, based on the evaluation of 7 specific queries.
- **02a_ecss_st_rag_expert-queries_v1.0.ipynb** Jupyter notebook to prompt expert queries.
- **02b_expert_eval_v1.0_anonym.ipynb** Evaluation of the expert queries Rag-LLM replies based on the completness score and correctness (see pptx/CAS_MAIN_TA_LN_ECSS-RAG_forDelivery.pdf for more information).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.zhaw.ch/bermic02/TA_LN_RAG_Local_LLM.git
    ```
2. Navigate to the project directory:
    ```bash
    cd TA_LN_RAG_Local_LLM
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## License

This project is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License. 

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

For more details, please visit [Creative Commons](https://creativecommons.org/licenses/by/4.0/).

