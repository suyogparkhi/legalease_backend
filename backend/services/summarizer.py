from fastapi import APIRouter, HTTPException, Request
import requests
import logging
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from backend.config.settings import get_settings
from backend.utils.pdf_utils import pdf_to_text

router = APIRouter(prefix="/summarizer", tags=["summarizer"])
settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ChatGroq(
    api_key=settings.API_KEY,
    model_name=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE
)
parser = StrOutputParser()

def generate_summary(text: str) -> str:
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n\n{text}"
    )

    chain = LLMChain(llm=model, prompt=prompt, output_parser=parser)
    result = chain.run(text=text)
    return result

@router.post("/")
async def summarize_pdf(request: Request):
    try:
        data = await request.json()
        file_url = data.get('docURL')
        
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
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")