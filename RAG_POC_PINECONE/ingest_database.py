import os
import pinecone
from dotenv import load_dotenv
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone  #  Correct Pinecone import
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec

#  Load Environment Variables
load_dotenv()

#  Pinecone API Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # Example: "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")  # Example: "documentanalyzer"
EMBEDDING_DIMENSION = 1536  #  Required for OpenAI embeddings

#  Initialize Pinecone Client (🔴 FIXED)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

#  Check if Pinecone index exists, if not create it
if PINECONE_INDEX_NAME not in [index_info.name for index_info in pc.list_indexes()]:
    print(f"🚀 Creating Pinecone index '{PINECONE_INDEX_NAME}' with dimension {EMBEDDING_DIMENSION}...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,  #  Required for vector search
        spec=ServerlessSpec(
            cloud="aws",  # Change based on your setup ("aws", "gcp", or "azure")
            region=PINECONE_ENV
        )
    )
else:
    print(f" Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

#  Load Embeddings Model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

#  Connect to Pinecone Vector Store (🔴 FIXED)
vector_store = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings_model,
    text_key="text"
)

#  Configuration for PDF Loading & Processing
DATA_PATH = "data"  # Folder containing PDFs

#  Load PDF Documents
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()
print(f" Loaded {len(raw_documents)} documents from {DATA_PATH}")

#  Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len
)
chunks = text_splitter.split_documents(raw_documents)
print(f" Created {len(chunks)} document chunks")

#  Generate Unique IDs for Chunks
uuids = [str(uuid4()) for _ in range(len(chunks))]

#  Store in Pinecone
vector_store.add_documents(documents=chunks, ids=uuids)
print(f" Successfully ingested {len(chunks)} document chunks into Pinecone!")