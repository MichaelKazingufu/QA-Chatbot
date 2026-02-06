from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()

# Load the document
raw_documents = PyPDFLoader("./test.pdf").load()

# Create embeddings model
embeddings = OllamaEmbeddings(model='mxbai-embed-large')

db_path = './chatbot_chroma_db'

split_documents = not os.path.exists(db_path) 

# Split the document into chunks if not done already

if split_documents :
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40
    )

    documents = text_splitter.split_documents(raw_documents)

#create the vectore store
vector_store = Chroma(
    collection_name="uafrika_status",
    persist_directory = db_path,
    embedding_function = embeddings
)

if split_documents:
    vector_store.add_documents(documents=documents)

#create the retriever
retriever = vector_store.as_retriever(
    search_kwargs={'k': 1}
)