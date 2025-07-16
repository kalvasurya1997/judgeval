import os
from dotenv import load_dotenv
load_dotenv()

from judgeval.tracer import Tracer, wrap
from openai import OpenAI
from agent.retriever import Retriever
from evaluation.judgeval_eval import run_evals
from monitoring.monitor import monitor_and_alert
from langchain_openai import OpenAIEmbeddings

# --- Pinecone v3 ---
from pinecone import Pinecone

# Load environment variables
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]
INDEX_NAME = "company-internal-documents"   # change to your real index name!
EMBED_DIM = 1024

# Judgeval API keys (make sure these are in your .env)
JUDGMENT_API_KEY = os.environ["JUDGMENT_API_KEY"]
JUDGMENT_ORG_ID = os.environ.get("JUDGMENT_ORG_ID", None)

# --- Judgeval Tracing ---
tracer = Tracer(project_name="research-rag-agent")

# --- OpenAI client wrapped for tracing ---
client = wrap(OpenAI())

# --- Pinecone v3 index + Retriever ---
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=EMBED_DIM)

@tracer.observe(span_type="tool")
def get_retriever():
    # Setup retriever using Pinecone index name
    return Retriever(INDEX_NAME, embeddings)

retriever = get_retriever()

@tracer.observe(span_type="agent")
def research_agent(question, retriever):
    # Retrieve context
    context_chunks = retriever.retrieve(question, top_k=5)
    context = "\n\n".join([c['text'] for c in context_chunks])
    prompt = f"""You are an AI assistant answering academic questions with citations from research papers.
Question: {question}

Relevant research context:
{context}

Your answer (cite sources if possible):
"""
    print("\n[AGENT] Sending prompt to OpenAI:")
    print(prompt[:400], "..." if len(prompt) > 400 else "")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print("\n[AGENT] Received answer:")
    print(answer[:400], "..." if len(answer) > 400 else "")
    return answer, context_chunks

# --- Main Q&A + Eval Loop ---
questions = [
    "What is transfer learning?",
    "Explain reinforcement learning.",
]

for q in questions:
    print(f"\nQ: {q}\n")
    answer, context_chunks = research_agent(q, retriever)
    print(f"\nA: {answer}\n")
    eval_results = run_evals(q, answer, context_chunks)
    monitor_and_alert(eval_results, q, answer)
