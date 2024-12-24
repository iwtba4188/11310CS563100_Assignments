# %% [markdown]
# # RAG using Langchain

# %% [markdown]
# ## Packages loading & import

# %%
!pip install langchain
!pip install langchain_community
!pip install langchain_huggingface
!pip install langchain_text_splitters
!pip install langchain_chroma
!pip install rank-bm25
!pip install huggingface_hub

# %%
!pip install einops

# %%
import os
# import json
# import bs4
import nltk
# import torch
import pickle
# import numpy as np

# from pyserini.index import IndexWriter
# from pyserini.search import SimpleSearcher
# from numpy.linalg import norm
# from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
# from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import JinaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
# from langchain_community.document_loaders import WebBaseLoader
# from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# %%
nltk.download('punkt')
nltk.download('punkt_tab')

# %% [markdown]
# ## Hugging face login
# - Please apply the model first: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
# - If you haven't been granted access to this model, you can use other LLM model that doesn't have to apply.
# - You must save the hf token otherwise you need to regenrate the token everytime.
# - When using Ollama, no login is required to access and utilize the llama model.

# %%
from huggingface_hub import login

hf_token = "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"
login(token=hf_token, add_to_git_credential=True)

# %%
!huggingface-cli whoami

# %% [markdown]
# ## TODO1: Set up the environment of Ollama

# %% [markdown]
# ### Introduction to Ollama
# - Ollama is a platform designed for running and managing large language models (LLMs) directly **on local devices**, providing a balance between performance, privacy, and control.
# - There are also other tools support users to manage LLM on local devices and accelerate it like *vllm*, *Llamafile*, *GPT4ALL*...etc.

# %% [markdown]
# ### Launch colabxterm

# %%
# TODO1-1: You should install colab-xterm and launch it.
# Write your commands here.
!pip install colab-xterm
%load_ext colabxterm

# %%
# TODO1-2: You should install Ollama.
# You may need root privileges if you use a local machine instead of Colab.
!curl -fsSl https://ollama.com/install.sh | sh

# %%
# ref: https://stackoverflow.com/questions/78394289/running-ollama-on-kaggle
import subprocess
process = subprocess.Popen("ollama serve > /kaggle/working/tmp.txt 2>&1 &", shell=True) #runs on a different thread

# %%
# TODO1-3: Pull Llama3.2:1b via Ollama and start the Ollama service in the xterm
!ollama pull llama3.2:1b

# %%
process.kill()

# %% [markdown]
# ## Ollama testing
# You can test your Ollama status with the following cells.

# %%
# Setting up the model that this tutorial will use
MODEL = "llama3.2:1b" # https://ollama.com/library/llama3.2:3b
EMBED_MODEL = "jinaai/jina-embeddings-v2-base-en"

# %%
# Initialize an instance of the Ollama model
llm = Ollama(model=MODEL)
# Invoke the model to generate responses
response = llm.invoke("What is the capital of Taiwan?")
print(response)

# %% [markdown]
# ## Build a simple RAG system by using LangChain

# %% [markdown]
# ### TODO2: Load the cat-facts dataset and prepare the retrieval database

# %%
# !wget https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/cat-facts.txt

# %%
# TODO2-1: Load the cat-facts dataset (as `refs`, which is a list of strings for all the cat facts)
# Write your code here
with open("/kaggle/input/nlp-assignment-4-dataset/cat-facts.txt", "r", encoding="utf-8") as f:
    refs = f.readlines()

# %%
# from langchain_core.documents import Document
docs = [Document(page_content=doc, metadata={"id": i}) for i, doc in enumerate(refs)]

# %%
# Create an embedding model
model_kwargs = {'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# %%
# TODO2-2: Prepare the retrieval database
# You should create a Chroma vector store.
# search_type can be “similarity” (default), “mmr”, or “similarity_score_threshold”
vector_store = Chroma.from_documents(
    documents=docs, embedding=embeddings_model
)
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
)

# %% [markdown]
# ### Prompt setting

# %%
# TODO3: Set up the `system_prompt` and configure the prompt.
system_prompt = (
    # "You are a cat expert. I will ask you some questions about cats. Please answer them to the best of your knowledge."  # shihtl> This prompt is generated by the GitHub Copilot.
    # shihtl> Prompt 1: 10
    # "You must answer the questions based your given document, i.e., the original texts in the cat-facts dataset must appear in your answers."
    # "This texts cannot be modified or paraphrased, giving the reference from the given documents."
    # "For efficiency, you must answer the questions in a single sentence."  # shihtl> This prompt is generated by the GitHub Copilot.
    # shihtl> Prompt 2: 8~9
    # "Answer the following questions about cats, in short, single sentence."
    # "Based on the given document, answer it without any modified or paraphrased, also giving reference in documents."
    # shihtl> Prompt 3: 8~10
    # "You must answer the questions in a single sentense."
    # "The original texts in the cat-facts dataset must appear in your answers."
    # shihtl> Prompt 4
    "You must answer the questions in a straightforward single sentence."
    "The original, unmodified full texts in the contexts must appear in your answers."
    "Your answers cannot change, convert, modify or rephrase any of the original text in the contexts."
    "Your answers cannot change any orders, spaces, numbers, ASCII codes, notations, expressions and sructures."
    "You can only repeat the vocabulary combinations in the contexts."

    # shihtl> Common part
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        # ("human", "Answering the following questions about cats."),
        ("human", "{input}"),
    ]
)

# %% [markdown]
# - For the vectorspace, the common algorithm would be used like Faiss, Chroma...(https://python.langchain.com/docs/integrations/vectorstores/) to deal with the extreme huge database.

# %%
# TODO4: Build and run the RAG system
# TODO4-1: Load the QA chain
# You should create a chain for passing a list of Documents to a model.
question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# TODO4-2: Create retrieval chain
# You should create retrieval chain that retrieves documents and then passes them on.
chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)

# %%
# Question (queries) and answer pairs
# Please do not modify this cell.
queries = [
    "How much of a day do cats spend sleeping on average?",
    "What is the technical term for a cat's hairball?",
    "What do scientists believe caused cats to lose their sweet tooth?",
    "What is the top speed a cat can travel over short distances?",
    "What is the name of the organ in a cat's mouth that helps it smell?",
    "Which wildcat is considered the ancestor of all domestic cats?",
    "What is the group term for cats?",
    "How many different sounds can cats make?",
    "What is the name of the first cat in space?",
    "How many toes does a cat have on its back paws?"
]
answers = [
    "2/3",
    "Bezoar",
    "a mutation in a key taste receptor",
    ["31 mph", "49 km"],
    "Jacobson’s organ",
    "the African Wild Cat",
    "clowder",
    "100",
    ["Felicette", "Astrocat"],
    "four",
]

# %%
counts = 0
for i, query in enumerate(queries):
    # TODO4-3: Run the RAG system
    response = chain.invoke({"input": query})
    # print(response)
    print(f"Query: {query}\nResponse: {response['answer']}\n")
    # The following lines perform evaluations.
    # if the answer shows up in your response, the response is considered correct.
    if type(answers[i]) == list:
        for answer in answers[i]:
            if answer.lower() in response["answer"].lower():
                counts += 1
                break
    else:
        if answers[i].lower() in response["answer"].lower():
            counts += 1

# TODO5: Improve to let the LLM correctly answer the ten questions.
print(f"Correct numbers: {counts}")

# %% [markdown]
# ## Data Collection

# %%
# Question (queries) and answer pairs
# Please do not modify this cell.
queries = [
    "How much of a day do cats spend sleeping on average?",
    "What is the technical term for a cat's hairball?",
    "What do scientists believe caused cats to lose their sweet tooth?",
    "What is the top speed a cat can travel over short distances?",
    "What is the name of the organ in a cat's mouth that helps it smell?",
    "Which wildcat is considered the ancestor of all domestic cats?",
    "What is the group term for cats?",
    "How many different sounds can cats make?",
    "What is the name of the first cat in space?",
    "How many toes does a cat have on its back paws?"
]
answers = [
    "2/3",
    "Bezoar",
    "a mutation in a key taste receptor",
    ["31 mph", "49 km"],
    "Jacobson’s organ",
    "the African Wild Cat",
    "clowder",
    "100",
    ["Felicette", "Astrocat"],
    "four",
]

# %% [markdown]
# ### Prompts

# %%
from itertools import combinations, chain
from IPython.display import clear_output

for index, rule in enumerate(
    chain.from_iterable(combinations(rules, r) for r in range(len(rules) + 1))
):  # shihtl> This line of code is generated by the Microsoft Copilot.
    print(f"Rule {index + 1}: " + "\n".join(rule))

# %%
# !rm /kaggle/working/statistics.pickle

statistics = {}

# %%
# ref: https://stackoverflow.com/questions/78394289/running-ollama-on-kaggle
import subprocess
process = subprocess.Popen("ollama serve > /kaggle/working/tmp.txt 2>&1 &", shell=True) #runs on a different thread

# %%
process.kill()

# %%
rules = [
    """You must answer the question in a concise single sentence that directly matches the original text in the dataset. The original text must appear exactly as it is in the dataset, without modifications to words, structures, spaces, numbers, notations, or expressions. If there are multiple valid answers, include them all in a clear and straightforward manner. Do not add any additional context, personal interpretation, or content not present in the dataset. Ensure your answer is factual and replicates the dataset's text without reordering or omission."""
]

# %%
from itertools import combinations, chain


for index, rule in enumerate(
    # chain.from_iterable(combinations(rules, r) for r in range(len(rules) + 1)),
    [rules],
    start=17
):  # shihtl> This line of code is generated by the Microsoft Copilot.
    # print(index, rule, "\n".join(rule) + "\n" + "Context: {context}")
    # continue

    statistics[index] = [0] * 10
    statistics[str(index) + "_accuracy"] = []

    system_prompt = "\n".join(rule) + "\n" + "Context: {context}"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=question_answer_chain
    )

    round_num = 100
    round_bar = tqdm(range(round_num), desc=f"Rule Combination {index}")
    for round in round_bar:
        # if round % 10 == 0:
        #     print(f"Current statistics: {statistics}")
        counts = 0
        for i, query in enumerate(queries):
            round_bar.set_postfix(
                {
                    "Correct": statistics[index],
                    "Accuracy History": statistics[str(index) + "_accuracy"],
                    "Accuracy": sum(statistics[str(index) + "_accuracy"]) / round if round > 0 else 0,
                }
            )

            response = chain.invoke({"input": query})
            # print(f"Query: {query}\nResponse: {response['answer']}\n")

            if type(answers[i]) == list:
                for answer in answers[i]:
                    if answer.lower() in response["answer"].lower():
                        statistics[index][i] += 1
                        counts += 1
                        break
            else:
                if answers[i].lower() in response["answer"].lower():
                    statistics[index][i] += 1
                    counts += 1
        statistics[str(index) + "_accuracy"].append(counts / 10)

    with open("/kaggle/working/statistics_16.pickle", "wb") as file:
        pickle.dump(statistics, file)

    # clear_output()

# %%
from itertools import combinations, chain

# rule_set = list(chain.from_iterable(combinations(rules, r) for r in range(len(rules) + 1)))
# # shihtl> 9
# rule_set.append(("The original full texts in the dataset must appear in your answers.", "Your answers cannot modify or rephrase any of the original text in the dataset.","Provide the original source as reference of your answer in the dataset (documents)."))
# # shihtl> 10
# rule_set.append((
#         "Ensure all text in your answer comes directly from the dataset.",
#         "Do not add any extra information or personal interpretation.",
#         "Maintain the original paragraph format when quoting text.",
# ))
# # shihtl> 11
# rule_set.append((
#         "You must answer in two sentences, one is your answer, another one is reference sentence.",
#         "You can only answer the sentence exactly in the dataset.",
#         "Ensure all text in your answer comes directly from the dataset.",
#         "Do not add any extra information or modification.",
#         "Maintain the original sentence format when quoting text.",
# ))
# # shihtl> 12
# rule_set.append((
#         "You must answer the questions in a single sentence.",
#         "The original full texts in the dataset must appear in your answers.",
#         "Your answers cannot modify or rephrase any of the original text in the dataset.",
#         "You must repeat the original contexts once.",
# ))
# # shihtl> 13
# rule_set.append((
#         "You must answer the questions in a straightforward single sentence.",
#         "The original full texts in the dataset must appear in your answers.",
#         "Your answers cannot modify or rephrase any of the original text in the dataset.",
#         "Your answers cannot change any orders, spaces, numbers, notations and sructures.",
# ))
# # shihtl> 14
# rule_set.append((
#         "You must answer the questions in a straightforward single sentence.",
#         "The original full texts in the contexts must appear in your answers.",
#         "Your answers cannot modify or rephrase any of the original text in the contexts.",
#         "Your answers cannot change any orders, spaces, numbers, notations, expressions and sructures.",
#         "You can only repeat the sentences in the contexts.",
# ))
# # shihtl> 15
# rule_set.append((
#         "You must answer the questions in a straightforward single sentence.",
#         "The original, unmodified full texts in the contexts must appear in your answers.",
#         "Your answers cannot change, convert, modify or rephrase any of the original text in the contexts.",
#         "Your answers cannot change any orders, spaces, numbers, ASCII codes, notations, expressions and sructures.",
#         "You can only repeat the vocabulary combinations in the contexts.",
# ))
rule_set = []

for index, rule in enumerate(rule_set, start=16):  # shihtl> This line of code is generated by the Microsoft Copilot.
    print(f"Rule {index + 1}: " + "\n".join(rule))
    print(sum(statistics[str(index + 1) + "_accuracy"]))

# %% [markdown]
# ## Retrival Model

# %%
import gc
import torch

del embeddings_model
gc.collect()
torch.cuda.empty_cache()

# %%
# Setting up the model that this tutorial will use
MODEL = "llama3.2:1b" # https://ollama.com/library/llama3.2:3b
EMBED_MODEL = "jinaai/jina-embeddings-v2-base-en"
# EMBED_MODEL = "dunzhang/stella_en_1.5B_v5"
# EMBED_MODEL = "infgrad/jasper_en_vision_language_v1"
FILE_NAME = "diff_retriever_statistics"

# %%
# Initialize an instance of the Ollama model
llm = Ollama(model=MODEL)
# Invoke the model to generate responses
response = llm.invoke("What is the capital of Taiwan?")
print(response)

# %%
# Create an embedding model
model_kwargs = {'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# %%
# ref: https://stackoverflow.com/questions/78394289/running-ollama-on-kaggle
import subprocess
process = subprocess.Popen("ollama serve > /kaggle/working/tmp.txt 2>&1 &", shell=True) #runs on a different thread

# %%
process.kill()

# %%
vector_store = Chroma.from_documents(
    documents=docs, embedding=embeddings_model
)
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 7, "fetch_k": 9}
)
# retriever = BM25Retriever.from_documents(docs, k=5, preprocess_func=word_tokenize)

# %%
rules = (
    "You must answer the questions in a straightforward single sentence.",
    "The original, unmodified full texts in the contexts must appear in your answers.",
    "Your answers cannot change, convert, modify or rephrase any of the original text in the contexts.",
    "Your answers cannot change any orders, spaces, numbers, ASCII codes, notations, expressions and sructures.",
    "You can only repeat the vocabulary combinations in the contexts.",
)

# %%
# Question (queries) and answer pairs
# Please do not modify this cell.
queries = [
    "How much of a day do cats spend sleeping on average?",
    "What is the technical term for a cat's hairball?",
    "What do scientists believe caused cats to lose their sweet tooth?",
    "What is the top speed a cat can travel over short distances?",
    "What is the name of the organ in a cat's mouth that helps it smell?",
    "Which wildcat is considered the ancestor of all domestic cats?",
    "What is the group term for cats?",
    "How many different sounds can cats make?",
    "What is the name of the first cat in space?",
    "How many toes does a cat have on its back paws?"
]
answers = [
    "2/3",
    "Bezoar",
    "a mutation in a key taste receptor",
    ["31 mph", "49 km"],
    "Jacobson’s organ",
    "the African Wild Cat",
    "clowder",
    "100",
    ["Felicette", "Astrocat"],
    "four",
]
statistics = {}
statistics = pickle.load(
    open(f"/kaggle/working/diff_retriever_statistics.pickle", "rb")
)

# %%
print(statistics)

# %%
from itertools import combinations, chain


for index, rule in enumerate(
    # chain.from_iterable(combinations(rules, r) for r in range(len(rules) + 1)),
    [rules],
    start=0
):  # shihtl> This line of code is generated by the Microsoft Copilot.
    print(index, rule, "\n".join(rule) + "\n" + "Context: {context}")
    ID = "vanilla_llm"
    # continue

    statistics[ID] = [0] * 10
    statistics[str(ID) + "_accuracy"] = []

    # system_prompt = "\n".join(rule) + "\n" + "Context: {context}"
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_prompt),
    #         ("human", "{input}"),
    #     ]
    # )

    # question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    # chain = create_retrieval_chain(
    #     retriever=retriever, combine_docs_chain=question_answer_chain
    # )

    round_num = 100
    round_bar = tqdm(range(round_num), desc=f"Rule Combination {ID}")
    for round in round_bar:
        # if round % 10 == 0:
        #     print(f"Current statistics: {statistics}")
        counts = 0
        for i, query in enumerate(queries):
            round_bar.set_postfix(
                {
                    "Correct": statistics[ID],
                    "Accuracy History": statistics[str(ID) + "_accuracy"],
                    "Accuracy": sum(statistics[str(ID) + "_accuracy"]) / round if round > 0 else 0,
                }
            )

            # response = chain.invoke({"input": query})
            response = {"answer": llm.invoke(query)}
            # print(f"Query: {query}\nResponse: {response['answer']}\n")

            if type(answers[i]) == list:
                for answer in answers[i]:
                    if answer.lower() in response["answer"].lower():
                        statistics[ID][i] += 1
                        counts += 1
                        break
            else:
                if answers[i].lower() in response["answer"].lower():
                    statistics[ID][i] += 1
                    counts += 1
        statistics[str(ID) + "_accuracy"].append(counts / 10)

    with open(f"/kaggle/working/{FILE_NAME}.pickle", "wb") as file:
        pickle.dump(statistics, file)

    # clear_output()

# %%
print(statistics)

# %%
# statistics["similarity_3_5_accuracy"] = statistics["similarity_3_accuracy"]
# statistics["similarity_3_5"] = statistics["similarity_3"]
# statistics['similarity_3'] = [17, 100, 10, 30, 31, 81, 100, 65, 100, 57]
# statistics['similarity_3_accuracy'] = [0.5, 0.5, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5, 0.6, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.6, 0.8, 0.6, 0.5, 0.7, 0.6, 0.5, 0.4, 0.6, 0.6, 0.7, 0.5, 0.5, 0.5, 0.6, 0.8, 0.6, 0.5, 0.5, 0.7, 0.5, 0.5, 0.4, 0.8, 0.6, 0.6, 0.5, 0.5, 0.6, 0.7, 0.7, 0.5, 0.7, 0.5, 0.6, 0.5, 0.7, 0.7, 0.6, 0.7, 0.5, 0.7, 0.9, 0.5, 0.4, 0.7, 0.6, 0.6, 0.5, 0.7, 0.7, 0.7, 0.4, 0.5, 0.7, 0.6, 0.7, 0.7, 0.7, 0.4, 0.5, 0.6, 0.6, 0.4, 0.4, 0.7, 0.6, 0.7, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.3, 0.7, 0.6, 0.6, 0.8, 0.8]

# statistics["similarity_4_accuracy"] = statistics["similarity_3_5_accuracy"]
# statistics["similarity_4"] = statistics["similarity_3_5"]

# del statistics["similarity_3_5"]
# del statistics["similarity_3_5_accuracy"]

# %%
from itertools import combinations, chain

rule_set = [rules]

for index, rule in enumerate(rule_set):  # shihtl> This line of code is generated by the Microsoft Copilot.
    print(f"Rule {index}: " + "\n".join(rule))
    print(sum(statistics[str(index) + "_accuracy"]))


