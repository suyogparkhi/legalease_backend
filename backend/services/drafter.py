from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from fastapi.responses import FileResponse
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from docx import Document
from reportlab.pdfgen import canvas
from backend.config.settings import get_settings
import tempfile
import os

router = APIRouter(prefix="/drafter", tags=["drafter"])
settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ChatGroq(
    api_key=settings.API_KEY,
    model_name=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE
)
parser = StrOutputParser()

@router.post("/generate_document")
async def generate_document(request: Dict[str, Any]) -> Dict[str, str]:
    try:
        case_details = request.get("case_details", "")
        ipc_sections = request.get("ipc_sections", [])

        if not case_details or not ipc_sections:
            raise HTTPException(status_code=400, detail="Missing case details or IPC sections")

        logger.info("Generating legal document")
        prompt = PromptTemplate(
            input_variables=["case_details", "ipc_sections"],
            template="""Draft a legal document for the following case:

Case Details: {case_details}

Relevant IPC Sections: {ipc_sections}

Please provide a well-structured legal document based on Indian laws and the given IPC sections."""
        )
        
        chain = LLMChain(llm=model, prompt=prompt, output_parser=parser)
        result = chain.run(case_details=case_details, ipc_sections=", ".join(ipc_sections))
        
        return {"document_content": result}
    except Exception as e:
        logger.error(f"Error generating document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating document: {str(e)}")

@router.post("/edit_document")
async def edit_document(request: Dict[str, str]) -> Dict[str, str]:
    try:
        document_content = request.get("document_content", "")

        if not document_content:
            raise HTTPException(status_code=400, detail="Missing document content")

        return {"document_content": document_content}
    except Exception as e:
        logger.error(f"Error editing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error editing document: {str(e)}")

@router.post("/export_document")
async def export_document(request: Dict[str, str], format: str) -> FileResponse:
    """
    Export document content to PDF or DOCX format.
    
    Args:
        request: Dictionary containing 'document_content' key
        format: Export format ('pdf' or 'docx')
    
    Returns:
        FileResponse with the generated document
        
    Raises:
        HTTPException: For invalid input or processing errors
    """
    try:
        # Validate input
        document_content = request.get("document_content")
        if not document_content:
            raise HTTPException(
                status_code=400,
                detail="Missing document content"
            )
        
        if format not in ["pdf", "docx"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported format. Use 'pdf' or 'docx'."
            )

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{format}"
        ) as tmp_file:
            tmp_path = tmp_file.name
            
            logger.info(f"Exporting document as {format}")
            
            if format == "pdf":
                # Generate PDF
                c = canvas.Canvas(tmp_path)
                # Add proper text wrapping and formatting
                y_position = 750
                for line in document_content.split('\n'):
                    c.drawString(100, y_position, line)
                    y_position -= 15
                c.save()
                
            else:
                # Generate DOCX
                doc = Document()
                doc.add_paragraph(document_content)
                doc.save(tmp_path)

            # Return file response
            return FileResponse(
                path=tmp_path,
                filename=f"document.{format}",
                media_type=f"application/{'pdf' if format == 'pdf' else 'vnd.openxmlformats-officedocument.wordprocessingml.document'}",
                background=None  # Process file deletion after response is sent
            )

    except Exception as e:
        logger.error(f"Error exporting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting document: {str(e)}"
        )
    
    finally:
        # Clean up temporary file if it exists
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")