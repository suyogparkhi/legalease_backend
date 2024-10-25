from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import logging
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import Optional, List
from backend.config.settings import get_settings

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ChatGroq(
    api_key=settings.API_KEY,
    model_name=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE
)

store = {}

class ChatMessage(BaseModel):
    sender: str
    content: str

class QueryRequest(BaseModel):
    query: str
    session_id: str
    new_chat: bool = False

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)

@router.post("/process_query")
def process_legal_query(request: QueryRequest):
    logger.info(f"Processing query: {request.query} for session: {request.session_id}")

    if not request.session_id:
        logger.error("Session ID is missing.")
        raise HTTPException(status_code=400, detail="Session ID is required.")

    if request.new_chat:
        request.session_id = str(uuid.uuid4())
        store[request.session_id] = InMemoryChatMessageHistory()
        logger.info(f"Started new chat with session ID: {request.session_id}")

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""You are an Indian legal assistant chatbot. Based on the following legal question, give a human-like response:
        
        Question: {query}
        
        Response:
        """
    )

    final_prompt = prompt_template.format(query=request.query)
    config = {"configurable": {"session_id": request.session_id}}

    try:
        session_history = get_session_history(request.session_id)
        session_history.add_user_message(request.query)

        response = with_message_history.invoke(
            [HumanMessage(content=final_prompt)],
            config=config,
        )
        
        logger.info(f"Model response: {response.content}")
        return {"response": response.content, "session_id": request.session_id}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.post("/new_chat")
def start_new_chat():
    session_id = str(uuid.uuid4())
    logger.info(f"New chat started with session ID: {session_id}")
    store[session_id] = InMemoryChatMessageHistory()
    return {"session_id": session_id}

@router.post("/save_chat/{session_id}")
def save_chat(session_id: str, chat_history: List[ChatMessage]):
    if session_id not in store:
        logger.warning(f"Session ID {session_id} not found. Creating new history.")

    new_history = InMemoryChatMessageHistory()
    for message in chat_history:
        if message.sender == "user":
            new_history.add_user_message(message.content)
        else:
            new_history.add_ai_message(message.content)

    store[session_id] = new_history
    logger.info(f"Chat history saved for session ID: {session_id}")
    return {"message": "Chat history saved successfully"}

@router.get("/get_chat_history/{session_id}")
def get_chat_history(session_id: str):
    if session_id not in store:
        logger.warning(f"Chat history requested for non-existent session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Session ID not found.")

    chat_history = store[session_id].messages
    logger.info(f"Retrieved chat history for session ID: {session_id}")
    return {"chat_history": chat_history}