from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer
from judgeval import JudgmentClient
import uuid

def run_evals(question, answer, context_chunks):
    context = [c['text'] for c in context_chunks]
    example = Example(
        input=question,
        actual_output=answer,
        retrieval_context=context
    )
    scorers = [
        FaithfulnessScorer(threshold=0.5)
        # HallucinationScorer(threshold=0.5)  # comment this out for now
    ]
    client = JudgmentClient()
    eval_run_name = str(uuid.uuid4())  # Always unique!
    results = client.assert_test(
        examples=[example],
        scorers=scorers,
        model="gpt-3.5-turbo",
        eval_run_name=eval_run_name
    )
    print("Evaluation done. Faithfulness check included.")
    return results or []
