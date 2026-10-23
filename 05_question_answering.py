# %% [markdown]
# Source notebook: 05_question_answering.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Question Answering

# %% [markdown]
# ## Overview
# 
# Recall the overall workflow for retrieval augmented generation (RAG):

# %% [markdown]
# ![overview.jpeg](attachment:overview.jpeg)

# %% [markdown]
# We discussed `Document Loading` and `Splitting` as well as `Storage` and `Retrieval`.
# 
# Let's load our vectorDB. 

# %%
# Notebook cell 5
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# %% [markdown]
# The code below was added to assign the openai LLM version filmed until it is deprecated, currently in Sept 2023. 
# LLM responses can often vary, but the responses may be significantly different when using a different model version.

# %%
# Notebook cell 7
import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

# %%
# Notebook cell 8
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# %%
# Notebook cell 9
print(vectordb._collection.count())

# %%
# Notebook cell 10
question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

# %%
# Notebook cell 11
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# %% [markdown]
# ### RetrievalQA chain

# %%
# Notebook cell 13
from langchain.chains import RetrievalQA

# %%
# Notebook cell 14
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

# %%
# Notebook cell 15
result = qa_chain({"query": question})

# %%
# Notebook cell 16
result["result"]

# %% [markdown]
# ### Prompt

# %%
# Notebook cell 18
from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# %%
# Notebook cell 19
# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# %%
# Notebook cell 20
question = "Is probability a class topic?"

# %%
# Notebook cell 21
result = qa_chain({"query": question})

# %%
# Notebook cell 22
result["result"]

# %%
# Notebook cell 23
result["source_documents"][0]

# %% [markdown]
# ### RetrievalQA chain types

# %%
# Notebook cell 25
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)

# %%
# Notebook cell 26
result = qa_chain_mr({"query": question})

# %%
# Notebook cell 27
result["result"]

# %% [markdown]
# If you wish to experiment on the `LangSmith platform` (previously known as LangChain Plus):
# 
#  * Go to [LangSmith](https://www.langchain.com/langsmith) and sign up
#  * Create an API key from your account's settings
#  * Use this API key in the code below   
#  * uncomment the code  
#  Note, the endpoint in the video differs from the one below. Use the one below.

# %%
# Notebook cell 29
#import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
#os.environ["LANGCHAIN_API_KEY"] = "..." # replace dots with your api key

# %%
# Notebook cell 30
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
result["result"]

# %%
# Notebook cell 31
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
result["result"]

# %% [markdown]
# ### RetrievalQA limitations
#  
# QA fails to preserve conversational history.

# %%
# Notebook cell 33
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

# %%
# Notebook cell 34
question = "Is probability a class topic?"
result = qa_chain({"query": question})
result["result"]

# %%
# Notebook cell 35
question = "why are those prerequesites needed?"
result = qa_chain({"query": question})
result["result"]

# %% [markdown]
# Note, The LLM response varies. Some responses **do** include a reference to probability which might be gleaned from referenced documents. The point is simply that the model does not have access to past questions or answers, this will be covered in the next section.

# %%
# Notebook cell 37

