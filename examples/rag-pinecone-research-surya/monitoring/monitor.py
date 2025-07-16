import json
import os
from datetime import datetime

def monitor_and_alert(eval_results, question, answer, log_path="monitoring/monitoring_log.jsonl"):
    """
    Monitor Judgeval evaluation results, alert for poor faithfulness or hallucination, and log results.
    """
    # Defensive check to avoid iterating over None
    if not eval_results:
        print("No evaluation results to monitor.")
        return

    # Ensure monitoring directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    timestamp = datetime.now().isoformat()
    for result in eval_results:
        scorer = result.get("scorer", "")
        score = result.get("score", 1.0)
        # Customizable alerting logic
        if scorer == "HallucinationScorer" and score > 0.5:
            print(f"\n ALERT: Hallucination detected for: {question}\nAnswer: {answer}\nScore: {score}\n")
        if scorer == "FaithfulnessScorer" and score < 0.5:
            print(f"\n ALERT: Low faithfulness for: {question}\nAnswer: {answer}\nScore: {score}\n")
        # Add timestamp, question, and answer to log
        result_log = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            **result
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_log, ensure_ascii=False) + "\n")
