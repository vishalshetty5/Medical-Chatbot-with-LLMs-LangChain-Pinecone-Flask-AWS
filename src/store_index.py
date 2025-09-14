from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os 
from typing import List
from langchain.schema import Document
from pinecone import Pinecone 
from pinecone import ServerlessSpec 
from src.helper import load_pdf_files, text_split, download_embeddings
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def filter_to_minimal_docs(docs: List[Document], min_length: int = 100) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs

minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunk = text_split(minimal_docs)  
embedding = download_embeddings()
index_name = "medical-chatbot"  # changed underscore to hyphen

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk, 
    embedding=embedding,
    index_name=index_name
)
extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunk = text_split(minimal_docs)
embedding = download_embeddings()

pinecone_api_key = PINECONE_API_KEY 

pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"  # changed underscore to hyphen

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk, 
    embedding=embedding,
    index_name=index_name
)