from fastapi import APIRouter, HTTPException, Request
import requests
import logging
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_extraction_chain
from backend.config.settings import get_settings
from backend.utils.pdf_utils import pdf_to_text
from langchain_core.runnables import RunnablePassthrough

router = APIRouter(prefix="/summarizer", tags=["summarizer"])
settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the ChatGroq model
model = ChatGroq(
    api_key=settings.API_KEY,
    model_name=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE
)

def generate_summary(text: str) -> str:
    try:
        # Create prompt template
        prompt = PromptTemplate.from_template(
            "Summarize the following text:\n\n{text}\n\nSummary:"
        )
        
        # Create the chain using the new method
        chain = prompt | model | StrOutputParser()
        
        # Invoke the chain with the text
        result = chain.invoke({"text": text})
        
        return result
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise

@router.post("/")
async def summarize_pdf(request: Request):
    try:
        # Parse request body
        data = await request.json()
        file_url = data.get('pdf_url')  # Changed from docURL to pdf_url to match frontend
        
        if not file_url:
            raise HTTPException(status_code=400, detail="File URL is required")

        logger.info(f"Fetching PDF from URL: {file_url}")
        pdf_response = requests.get(file_url)
        pdf_response.raise_for_status()

        logger.info("Converting PDF to text")
        pdf_text = pdf_to_text(pdf_response.content)
        
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="PDF contains no text")

        logger.info("Generating summary")
        summary = generate_summary(pdf_text)
        
        return {"summary": summary}
    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise HTTPException(status_code=400, detail=f"Error retrieving the PDF file: {str(http_err)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))