from rag_chain import answer_question
from hallucination_check import check_hallucination
from dotenv import load_dotenv
load_dotenv()

EVAL_DATASET = [
    {"question": "Where is the Eiffel Tower?", "ground_truth": "The Eiffel Tower is in Paris, France."},
    {"question": "Who created Python?", "ground_truth": "Python was created by Guido van Rossum."},
    {"question": "What is LangChain?", "ground_truth": "LangChain is a framework for building LLM applications."},
]

def run_evaluation():
    results = []
    for item in EVAL_DATASET:
        answer = answer_question(item["question"])
        halluc = check_hallucination(item["ground_truth"], answer)
        result = {"question": item["question"], "answer": answer, "ground_truth": item["ground_truth"], **halluc}
        results.append(result)
        print(f"Q: {item['question']}")
        print(f"A: {answer}")
        print(f"Grounded: {halluc['is_grounded']} | Score: {halluc['similarity_score']}")
        print("---")
    return results

if __name__ == "__main__":
    run_evaluation()