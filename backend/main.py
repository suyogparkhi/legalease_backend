from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.services import chatbot, summarizer, doc_qna, drafter
from backend.config.settings import get_settings

app = FastAPI(title="Legal AI Services")

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

app.include_router(chatbot.router)
app.include_router(summarizer.router)
app.include_router(doc_qna.router)
app.include_router(drafter.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)