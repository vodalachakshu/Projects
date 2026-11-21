from fastapi import FastAPI

app = FastAPI(title="Document Q&A Service", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}
