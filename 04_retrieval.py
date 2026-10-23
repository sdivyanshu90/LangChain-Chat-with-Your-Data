# %% [markdown]
# Source notebook: 04_retrieval.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Retrieval
# 
# Retrieval is the centerpiece of our retrieval augmented generation (RAG) flow. 
# 
# Let's get our vectorDB from before.

# %% [markdown]
# ## Vectorstore retrieval

# %%
# Notebook cell 3
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# %%
# Notebook cell 4
#!pip install lark

# %% [markdown]
# ### Similarity Search

# %%
# Notebook cell 6
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

# %%
# Notebook cell 7
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# %%
# Notebook cell 8
print(vectordb._collection.count())

# %%
# Notebook cell 9
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

# %%
# Notebook cell 10
smalldb = Chroma.from_texts(texts, embedding=embedding)

# %%
# Notebook cell 11
question = "Tell me about all-white mushrooms with large fruiting bodies"

# %%
# Notebook cell 12
smalldb.similarity_search(question, k=2)

# %%
# Notebook cell 13
smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3)

# %% [markdown]
# ### Addressing Diversity: Maximum marginal relevance
# 
# Last class we introduced one problem: how to enforce diversity in the search results.
#  
# `Maximum marginal relevance` strives to achieve both relevance to the query *and diversity* among the results.

# %%
# Notebook cell 15
question = "what did they say about matlab?"
docs_ss = vectordb.similarity_search(question,k=3)

# %%
# Notebook cell 16
docs_ss[0].page_content[:100]

# %%
# Notebook cell 17
docs_ss[1].page_content[:100]

# %% [markdown]
# Note the difference in results with `MMR`.

# %%
# Notebook cell 19
docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)

# %%
# Notebook cell 20
docs_mmr[0].page_content[:100]

# %%
# Notebook cell 21
docs_mmr[1].page_content[:100]

# %% [markdown]
# ### Addressing Specificity: working with metadata
# 
# In last lecture, we showed that a question about the third lecture can include results from other lectures as well.
# 
# To address this, many vectorstores support operations on `metadata`.
# 
# `metadata` provides context for each embedded chunk.

# %%
# Notebook cell 23
question = "what did they say about regression in the third lecture?"

# %%
# Notebook cell 24
docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source":"docs/cs229_lectures/MachineLearning-Lecture03.pdf"}
)

# %%
# Notebook cell 25
for d in docs:
    print(d.metadata)

# %%
# Notebook cell 26


# %% [markdown]
# ### Addressing Specificity: working with metadata using self-query retriever
# 
# But we have an interesting challenge: we often want to infer the metadata from the query itself.
# 
# To address this, we can use `SelfQueryRetriever`, which uses an LLM to extract:
#  
# 1. The `query` string to use for vector search
# 2. A metadata filter to pass in as well
# 
# Most vector databases support metadata filters, so this doesn't require any new databases or indexes.

# %%
# Notebook cell 28
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# %%
# Notebook cell 29
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

# %% [markdown]
# **Note:** The default model for `OpenAI` ("from langchain.llms import OpenAI") is `text-davinci-003`. Due to the deprication of OpenAI's model `text-davinci-003` on 4 January 2024, you'll be using OpenAI's recommended replacement model `gpt-3.5-turbo-instruct` instead.

# %%
# Notebook cell 31
document_content_description = "Lecture notes"
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# %%
# Notebook cell 32
question = "what did they say about regression in the third lecture?"

# %% [markdown]
# **You will receive a warning** about predict_and_parse being deprecated the first time you executing the next line. This can be safely ignored.

# %%
# Notebook cell 34
docs = retriever.get_relevant_documents(question)

# %%
# Notebook cell 35
for d in docs:
    print(d.metadata)

# %% [markdown]
# ### Additional tricks: compression
# 
# Another approach for improving the quality of retrieved docs is compression.
# 
# Information most relevant to a query may be buried in a document with a lot of irrelevant text. 
# 
# Passing that full document through your application can lead to more expensive LLM calls and poorer responses.
# 
# Contextual compression is meant to fix this. 

# %%
# Notebook cell 37
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# %%
# Notebook cell 38
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# %%
# Notebook cell 39
# Wrap our vectorstore
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
compressor = LLMChainExtractor.from_llm(llm)

# %%
# Notebook cell 40
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)

# %%
# Notebook cell 41
question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)

# %% [markdown]
# ## Combining various techniques

# %%
# Notebook cell 43
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr")
)

# %%
# Notebook cell 44
question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)

# %% [markdown]
# ## Other types of retrieval
# 
# It's worth noting that vectordb as not the only kind of tool to retrieve documents. 
# 
# The `LangChain` retriever abstraction includes other ways to retrieve documents, such as TF-IDF or SVM.

# %%
# Notebook cell 46
from langchain.retrievers import SVMRetriever
from langchain.retrievers import TFIDFRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# %%
# Notebook cell 47
# Load PDF
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
splits = text_splitter.split_text(joined_page_text)

# %%
# Notebook cell 48
# Retrieve
svm_retriever = SVMRetriever.from_texts(splits,embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

# %%
# Notebook cell 49
question = "What are major topics for this class?"
docs_svm=svm_retriever.get_relevant_documents(question)
docs_svm[0]

# %%
# Notebook cell 50
question = "what did they say about matlab?"
docs_tfidf=tfidf_retriever.get_relevant_documents(question)
docs_tfidf[0]

# %%
# Notebook cell 51

