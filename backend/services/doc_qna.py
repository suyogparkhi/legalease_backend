from fastapi import APIRouter, File, Form, UploadFile, HTTPException
import io
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

router = APIRouter(prefix="/doc-qna", tags=["doc-qna"])
settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdf_and_ask_question(pdf_file: bytes, question: str):
    logger.info("Processing PDF and generating answer")
    

    try:
        with io.BytesIO(pdf_file) as temp_pdf:
            file_loader = PyPDFLoader(temp_pdf)
            pages = file_loader.load_and_split()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        docs = splitter.split_documents(pages)

        vector_storage = FAISS.from_documents(docs, FakeEmbeddings(size=1352))
        retriever = vector_storage.as_retriever()

        question_template = """
        You are a Smart Bot that answers questions based on the context given to you.
        Return the answer and don't make up things on your own.
        context: {context}
        question: {question}
        """
        prompt = PromptTemplate.from_template(template=question_template)

        parser = StrOutputParser()
        groq_model = ChatGroq(
            api_key=settings.API_KEY,
            model_name=settings.MODEL_NAME,
            temperature=settings.TEMPERATURE
        )

        result = RunnableParallel(
            context=retriever,
            question=RunnablePassthrough()
        )
        chain = result | prompt | groq_model | parser

        answer = chain.invoke(question)
        logger.info("Successfully generated answer")
        return answer

    except Exception as e:
        logger.error(f"Error processing PDF and generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/ask_query")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        file_bytes = await file.read()
        answer = process_pdf_and_ask_question(file_bytes, question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")