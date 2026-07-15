# llm-eval-pipeline
An end-to-end LLM evaluation pipeline that logs model runs, scores outputs against ground truth benchmarks, and flags hallucinated responses using semantic grounding checks.

## What it does

- Runs a RAG pipeline using Groq (Llama 3.3) as the LLM
- Evaluates model outputs against ground truth using TF-IDF similarity scoring
- Detects hallucinated entities using spaCy NER
- Flags responses as grounded or ungrounded with a confidence score
- Traces runs via LangSmith for observability

## Tech Stack

Python · LangChain · LangSmith · Groq (Llama 3.3) · spaCy · TF-IDF · FAISS · Pytest

## Sample Output
Q: Where is the Eiffel Tower?
A: The Eiffel Tower is located in Paris, France.
Grounded: True | Score: 0.883
Q: Who created Python?
A: Guido van Rossum created Python in 1991.
Grounded: False | Score: 0.559
Q: What is LangChain?
A: LangChain is a framework for building LLM applications.
Grounded: True | Score: 0.736

## How to run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python eval_pipeline.py
pytest tests/ -v
```



## Relevance

Mirrors production LLM QA workflows — hallucination detection, output scoring, and observability are core requirements in enterprise AI evaluation pipelines.
