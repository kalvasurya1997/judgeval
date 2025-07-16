import glob
import os
import json
import tqdm
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings

# ---- Pinecone v3 imports ----
from pinecone import Pinecone, ServerlessSpec

# ---- Load .env ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'))

PDF_FOLDER = os.path.join(PROJECT_ROOT, "data", "papers")
CHUNK_FILE = os.path.join(PROJECT_ROOT, "data", "chunks.json")
INDEX_NAME = "company-internal-documents"  # <-- your real index name!
EMBED_DIM = 1024
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

def chunk_pdfs():
    pdf_files = glob.glob(f"{PDF_FOLDER}/*.pdf")
    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(chunks)} chunks from PDFs.")
    # Save as JSON
    with open(CHUNK_FILE, "w", encoding="utf-8") as out:
        json.dump(
            [
                {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ], out, ensure_ascii=False, indent=2
        )
    return chunks

def embed_and_upsert(chunks):
    # ---- Pinecone v3 setup ----
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=EMBED_DIM)
    batch_size = 8
    for i in tqdm.tqdm(range(0, len(chunks), batch_size)):
        chunk_batch = chunks[i:i+batch_size]
        texts = [doc.page_content for doc in chunk_batch]
        vectors = embeddings.embed_documents(texts)
        pinecone_vectors = []
        for j, doc in enumerate(chunk_batch):
            meta = {
                "source": doc.metadata.get("source", ""),
                "chunk_id": str(i+j),
                "content": doc.page_content
            }
            pinecone_vectors.append(
                {
                    "id": f"doc-{i+j}",
                    "values": vectors[j],
                    "metadata": meta
                }
            )
        index.upsert(vectors=pinecone_vectors, namespace="default")

if __name__ == "__main__":
    chunks = chunk_pdfs()
    embed_and_upsert(chunks)
    print("All chunks embedded and upserted to Pinecone.")
