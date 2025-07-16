from judgeval.tracer import Tracer
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings  # Use the new package!
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
tracer = Tracer(project_name="research-rag-agent")

class Retriever:
    def __init__(self, index_name, embedding):
        # Setup Pinecone v3 client and get index handle
        api_key = os.environ["PINECONE_API_KEY"]
        pc = Pinecone(api_key=api_key)
        self.index = pc.Index(index_name)
        # Setup LangChain vectorstore wrapper
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=embedding,
            text_key="content",
            namespace="default"
        )

    @tracer.observe(span_type="tool")
    def retrieve(self, query, top_k=5):
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "id": doc.metadata.get("chunk_id", "")
            } for doc in docs
        ]
