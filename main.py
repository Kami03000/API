from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from scidownl import scihub_download
from pydantic import BaseModel
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
import base64
import logging
from openai import OpenAI
import asyncio
import os
import re
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Ensure directories exist
os.makedirs("downloads", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Mount static directories
app.mount("/downloads", StaticFiles(directory="downloads"), name="downloads")
app.mount("/images", StaticFiles(directory="images"), name="images")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.kmizeolite.com"],
    # allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class OpenAIKeyRequest(BaseModel):
    openai_api_key: str

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def extract_pdf_metadata(pdf_file):
    metadata = {}
    try:
        with open(pdf_file, 'rb') as f:
            parser = PDFParser(f)
            document = PDFDocument(parser)
            if document.info:
                metadata = document.info[0]
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
    return metadata

def clean_text(text):
    try:
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)
        text = re.sub(r'(\s{2,}|\t)', ' ', text)
        text = re.sub(r'(Figure \d+:|Table \d+:)', '', text, flags=re.IGNORECASE)
        return text.strip()
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return text

def extract_pdf_text_by_page(pdf_file):
    try:
        text = extract_text(pdf_file)
        pages = text.split('\f')
        return [clean_text(page) for page in pages[:4] if page.strip()]
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return []

def process_pdf(pdf_file):
    metadata = extract_pdf_metadata(pdf_file)
    text_pages = extract_pdf_text_by_page(pdf_file)
    return [Document(page_content=page, metadata=metadata) for page in text_pages]

def get_summary(doc_objs, openai_api_key: str):
    prompt_template = """Please analyze the following text and provide the following details:
    1.Clearly identify and state the title of the article or chapter.
    2. Write a comprehensive, and coherent summary that thoroughly captures the main points but not that much lenghty, key insights, supporting details, and any critical arguments presented in the text. The summary should ensure no significant information is missed and should be proportionate to the text length.
    3. Explain the relevance and significance of the article, including its importance to its field or topic, how it contributes to ongoing discussions, and the benefits it offers to readers.
    4. Extract keywords if keywords of the text.
    Analyze the following text \n {text}:
    CONCISE SUMMARY:"""

    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=openai_api_key)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    return stuff_chain.run(doc_objs)

@app.post("/download/")
async def download(request: Request):
    try:
        files = os.listdir('downloads/')
        for f in files:
            os.remove(os.path.join('downloads', f))

        data = await request.json()
        doi = data.get("doi")
        save_path = data.get("path")

        if not doi:
            raise HTTPException(status_code=400, detail="DOI not provided")

        if not save_path:
            save_path = os.path.join("downloads", "default.pdf")
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # scihub_download always puts in 'downloads' folder
        scihub_download(doi, out="downloads/", paper_type='doi')

        downloaded_files = glob.glob("downloads/*.pdf")
        if not downloaded_files:
            raise HTTPException(status_code=500, detail="No PDF found after download")

        return JSONResponse(content={"message": "Download attempted"}, status_code=200)

    except Exception as e:
        logging.error(f"Error downloading paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    openai_api_key: str = Form(...)
):
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        documents = process_pdf(file_path)
        summary = get_summary(documents, openai_api_key)

        os.remove(file_path)
        return JSONResponse(content={"summary": summary})

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image/")
async def generate_image(
    openai_api_key: str = Form(...),
    save_path: str = Form(...),
    title: str = Form(...)
):
    try:
        files = os.listdir('images/')
        for f in files:
            os.remove(os.path.join('images', f))

        clean_title = re.sub(r'[^\w\-_]', '_', title)[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png"
        full_path = os.path.join(save_path, filename)

        client = OpenAI(api_key=openai_api_key)
        prompt = f"Create a simple and natural image inspired by the title '{title}', but ensure there is no text or overlapping text on the image."

        result = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            response_format="b64_json",
            n=1,
            size="1024x1024"
        )

        image_data = base64.b64decode(result.data[0].b64_json)
        with open(full_path, 'wb') as f:
            f.write(image_data)

        return JSONResponse(content={
            "message": "Image generated and saved successfully",
            "path": full_path,
            "filename": filename
        })

    except Exception as e:
        logging.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8282)
