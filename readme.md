
# Langchain RAG Multimodal

This project demonstrates the use of a multimodal retrieval system that handles various document types including text, tables, and images. The system parses PDF documents, embeds these parsed elements using a multimodal AI, stores these embeddings into a database, and finally builds a Retrieval-Augmented Generation (RAG) retriever for querying.

## Features

1. Use Unstructured to parse PDF into text chunks, tables, and images.
2. Use multimodal AI to embed those parsed elements.
3. Store those elements' embeddings into the database.
4. Build RAG retriever.
5. Query the multimodal RAG retriever.

## How to Use

### System Requirements

- Operating System: Ubuntu 20.04

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hahahassan/Langchain_RAG_multimodal.git
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **API Key Configuration:**
   - Enter your API key in the `config.py` file.

### Running the Notebook

Open and run the provided Jupyter notebook (`rag_multimodal.ipynb`) to see the implementation and usage of the retrieval system.

```bash
jupyter notebook rag_multimodal.ipynb
```

## Example Query

To see an example of querying the retriever and displaying results, you can run the following:

```python
query = "What is a transformer?"
output = retrieve_and_answer_question(retriever, query)
```

## Additional Information

This repository uses the paper "Attention Is All You Need" as an example.



