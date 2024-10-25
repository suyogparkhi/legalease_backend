import fitz
from fastapi import HTTPException

def pdf_to_text(pdf_data: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")