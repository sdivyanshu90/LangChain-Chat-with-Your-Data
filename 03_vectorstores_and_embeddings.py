# %% [markdown]
# Source notebook: 03_vectorstores_and_embeddings.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

from IPython import get_ipython as _active_ipython

def get_ipython():
    """Use the active shell and expose this script namespace to line magics."""
    shell = _active_ipython()
    if shell is None:
        raise RuntimeError("This source uses IPython commands; run it with ipython instead of python.")
    import inspect
    caller = inspect.currentframe().f_back
    shell.user_ns.update(caller.f_globals)
    shell.user_ns.update(caller.f_locals)
    return shell

# %% [markdown]
# # Vectorstores and Embeddings
# 
# Recall the overall workflow for retrieval augmented generation (RAG):

# %% [markdown]
# ![overview.jpeg](attachment:overview.jpeg)

# %%
# Notebook cell 3
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# %% [markdown]
# We just discussed `Document Loading` and `Splitting`.

# %%
# Notebook cell 5
from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# %%
# Notebook cell 6
# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

# %%
# Notebook cell 7
splits = text_splitter.split_documents(docs)

# %%
# Notebook cell 8
len(splits)

# %% [markdown]
# ## Embeddings
# 
# Let's take our splits and embed them.

# %%
# Notebook cell 10
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

# %%
# Notebook cell 11
sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

# %%
# Notebook cell 12
embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

# %%
# Notebook cell 13
import numpy as np

# %%
# Notebook cell 14
np.dot(embedding1, embedding2)

# %%
# Notebook cell 15
np.dot(embedding1, embedding3)

# %%
# Notebook cell 16
np.dot(embedding2, embedding3)

# %% [markdown]
# ## Vectorstores

# %%
# Notebook cell 18
# ! pip install chromadb

# %%
# Notebook cell 19
from langchain.vectorstores import Chroma

# %%
# Notebook cell 20
persist_directory = 'docs/chroma/'

# %%
# Notebook cell 21
get_ipython().system('rm -rf ./docs/chroma  # remove old database files if any')

# %%
# Notebook cell 22
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

# %%
# Notebook cell 23
print(vectordb._collection.count())

# %% [markdown]
# ### Similarity Search

# %%
# Notebook cell 25
question = "is there an email i can ask for help"

# %%
# Notebook cell 26
docs = vectordb.similarity_search(question,k=3)

# %%
# Notebook cell 27
len(docs)

# %%
# Notebook cell 28
docs[0].page_content

# %% [markdown]
# Let's save this so we can use it later!

# %%
# Notebook cell 30
vectordb.persist()

# %% [markdown]
# ## Failure modes
# 
# This seems great, and basic similarity search will get you 80% of the way there very easily. 
# 
# But there are some failure modes that can creep up. 
# 
# Here are some edge cases that can arise - we'll fix them in the next class.

# %%
# Notebook cell 32
question = "what did they say about matlab?"

# %%
# Notebook cell 33
docs = vectordb.similarity_search(question,k=5)

# %% [markdown]
# Notice that we're getting duplicate chunks (because of the duplicate `MachineLearning-Lecture01.pdf` in the index).
# 
# Semantic search fetches all similar documents, but does not enforce diversity.
# 
# `docs[0]` and `docs[1]` are indentical.

# %%
# Notebook cell 35
docs[0]

# %%
# Notebook cell 36
docs[1]

# %% [markdown]
# We can see a new failure mode.
# 
# The question below asks a question about the third lecture, but includes results from other lectures as well.

# %%
# Notebook cell 38
question = "what did they say about regression in the third lecture?"

# %%
# Notebook cell 39
docs = vectordb.similarity_search(question,k=5)

# %%
# Notebook cell 40
for doc in docs:
    print(doc.metadata)

# %%
# Notebook cell 41
print(docs[4].page_content)

# %% [markdown]
# Approaches discussed in the next lecture can be used to address both!

# %%
# Notebook cell 43

