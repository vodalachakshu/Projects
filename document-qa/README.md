# Document Q&A Service

A grounded retrieval-augmented generation (RAG) API built with FastAPI.

## Features

- Upload and index TXT, PDF, DOCX, and Markdown documents
- Retrieve relevant document passages for a query
- Return answers grounded in source passages
- Include citations and confidence information
- Validate input and provide consistent JSON errors

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `/docs` for the interactive API documentation.
