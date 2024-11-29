from fastapi import APIRouter, File, Form, UploadFile, HTTPException
import tempfile
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from backend.config.settings import get_settings
import requests
from pydantic import BaseModel

router = APIRouter(prefix="/doc-qna", tags=["doc-qna"])
settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = StrOutputParser()
groq_model = ChatGroq(
    api_key=settings.API_KEY,
    model_name=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE
)

class QueryRequest(BaseModel):
    pdf_url: str
    question: str

def process_pdf_and_ask_question(pdf_url: str, question: str):
    logger.info("Downloading PDF from URL and generating answer")

    try:
        # Download the PDF file
        response = requests.get(pdf_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf_path = temp_pdf.name

        # Load and split the PDF
        file_loader = PyPDFLoader(temp_pdf_path)
        pages = file_loader.load_and_split()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        docs = splitter.split_documents(pages)

        # Create a vector storage and retriever
        vector_storage = FAISS.from_documents(docs, FakeEmbeddings(size=1352))
        retriever = vector_storage.as_retriever()

        # Define the question prompt
        question_template = """
        You are a Smart Bot that answers questions based on the context given to you.
        Return the answer and don't make up things on your own.
        context: {context}
        question: {question}
        """
        prompt = PromptTemplate.from_template(template=question_template)

        result = RunnableParallel(
            context=retriever,
            question=RunnablePassthrough()
        )
        chain = result | prompt | groq_model | parser

        # Generate the answer
        answer = chain.invoke(question)
        logger.info("Successfully generated answer")
        return answer

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Error downloading PDF: {str(req_err)}")
        raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(req_err)}")
    except Exception as e:
        logger.error(f"Error processing PDF and generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    

@router.post("/ask_query")
async def ask_question(payload: QueryRequest):
    try:
        # Process the PDF URL and generate the answer
        answer = process_pdf_and_ask_question(payload.pdf_url, payload.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
