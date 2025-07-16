from judgeval.tracer import Tracer
from openai import OpenAI
from agent.retriever import Retriever
import os

tracer = Tracer(project_name="research-rag-agent")

# Pass api_key only if not set via environment, else it picks up OPENAI_API_KEY
client = OpenAI()

@tracer.observe(span_type="agent")
def research_agent(question, retriever):
    # Retrieve relevant context chunks from Pinecone
    context_chunks = retriever.retrieve(question, top_k=5)
    context = "\n\n".join([c['text'] for c in context_chunks])

    prompt = f"""You are an AI assistant answering academic questions with citations from research papers.
Question: {question}

Relevant research context:
{context}

Your answer (cite sources if possible):
"""
    print("\n[AGENT] Sending prompt to OpenAI:")
    print(prompt[:500], "..." if len(prompt) > 500 else "")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print("\n[AGENT] Received answer:")
    print(answer[:500], "..." if len(answer) > 500 else "")
    return answer, context_chunks
