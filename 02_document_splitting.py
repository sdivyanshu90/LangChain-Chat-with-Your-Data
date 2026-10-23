# %% [markdown]
# Source notebook: 02_document_splitting.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Document Splitting

# %%
# Notebook cell 2
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# %%
# Notebook cell 3
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# %%
# Notebook cell 4
chunk_size =26
chunk_overlap = 4

# %%
# Notebook cell 5
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# %% [markdown]
# Why doesn't this split the string below?

# %%
# Notebook cell 7
text1 = 'abcdefghijklmnopqrstuvwxyz'

# %%
# Notebook cell 8
r_splitter.split_text(text1)

# %%
# Notebook cell 9
text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'

# %%
# Notebook cell 10
r_splitter.split_text(text2)

# %% [markdown]
# Ok, this splits the string but we have an overlap specified as 5, but it looks like 3? (try an even number)

# %%
# Notebook cell 12
text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"

# %%
# Notebook cell 13
r_splitter.split_text(text3)

# %%
# Notebook cell 14
c_splitter.split_text(text3)

# %%
# Notebook cell 15
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator = ' '
)
c_splitter.split_text(text3)

# %% [markdown]
# Try your own examples!

# %% [markdown]
# ## Recursive splitting details
# 
# `RecursiveCharacterTextSplitter` is recommended for generic text. 

# %%
# Notebook cell 18
some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

# %%
# Notebook cell 19
len(some_text)

# %%
# Notebook cell 20
c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""]
)

# %%
# Notebook cell 21
c_splitter.split_text(some_text)

# %%
# Notebook cell 22
r_splitter.split_text(some_text)

# %% [markdown]
# Let's reduce the chunk size a bit and add a period to our separators:

# %%
# Notebook cell 24
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""]
)
r_splitter.split_text(some_text)

# %%
# Notebook cell 25
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
r_splitter.split_text(some_text)

# %%
# Notebook cell 26
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()

# %%
# Notebook cell 27
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

# %%
# Notebook cell 28
docs = text_splitter.split_documents(pages)

# %%
# Notebook cell 29
len(docs)

# %%
# Notebook cell 30
len(pages)

# %%
# Notebook cell 31
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
notion_db = loader.load()

# %%
# Notebook cell 32
docs = text_splitter.split_documents(notion_db)

# %%
# Notebook cell 33
len(notion_db)

# %%
# Notebook cell 34
len(docs)

# %% [markdown]
# ## Token splitting
# 
# We can also split on token count explicity, if we want.
# 
# This can be useful because LLMs often have context windows designated in tokens.
# 
# Tokens are often ~4 characters.

# %%
# Notebook cell 36
from langchain.text_splitter import TokenTextSplitter

# %%
# Notebook cell 37
text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)

# %%
# Notebook cell 38
text1 = "foo bar bazzyfoo"

# %%
# Notebook cell 39
text_splitter.split_text(text1)

# %%
# Notebook cell 40
text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

# %%
# Notebook cell 41
docs = text_splitter.split_documents(pages)

# %%
# Notebook cell 42
docs[0]

# %%
# Notebook cell 43
pages[0].metadata

# %% [markdown]
# ## Context aware splitting
# 
# Chunking aims to keep text with common context together.
# 
# A text splitting often uses sentences or other delimiters to keep related text together but many documents (such as Markdown) have structure (headers) that can be explicitly used in splitting.
# 
# We can use `MarkdownHeaderTextSplitter` to preserve header metadata in our chunks, as show below.

# %%
# Notebook cell 45
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

# %%
# Notebook cell 46
markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Molly"""

# %%
# Notebook cell 47
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# %%
# Notebook cell 48
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)

# %%
# Notebook cell 49
md_header_splits[0]

# %%
# Notebook cell 50
md_header_splits[1]

# %% [markdown]
# Try on a real Markdown file, like a Notion database.

# %%
# Notebook cell 52
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
txt = ' '.join([d.page_content for d in docs])

# %%
# Notebook cell 53
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# %%
# Notebook cell 54
md_header_splits = markdown_splitter.split_text(txt)

# %%
# Notebook cell 55
md_header_splits[0]

# %%
# Notebook cell 56

