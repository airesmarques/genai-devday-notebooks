# MongoDB GenAI Developer Day Notebooks
This repository is home to all the Jupyter Notebooks required for MongoDB's GenAI Developer Day. 

The self-paced versions of the Developer Days workshops are linked below:

* [Vector Search: Beginner to Pro](https://mongodb-developer.github.io/vector-search-lab/)

* [Building RAG Applications with MongoDB](https://mongodb-developer.github.io/ai-rag-lab/)

* [The A to Z of Building AI Agents](https://mongodb-developer.github.io/ai-agents-lab/)
  
## CLI Application

This repository also includes a one-shot CLI for a Retrieval-Augmented Generation (RAG) flow.

1. Copy environment variables:

   cp .env.example .env

2. Install dependencies:

   pip install -r requirements.txt

3. Run the CLI:

   python rag_app.py "Your question here"

   To skip data ingestion & indexing (i.e. after the first run), add --skip-prepare:

      python rag_app.py --skip-prepare "Your question here"
