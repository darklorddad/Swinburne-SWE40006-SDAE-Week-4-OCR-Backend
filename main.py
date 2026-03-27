# main.py
import base64
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="OCR API")

# Update origins to include your Vercel URL after deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ocr-frontend.vercel.app",     # your Vercel URL
        "https://ocr-frontend-*.vercel.app",   # preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
) if api_key else None


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    if not client:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured on the server")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    b64_image = base64.b64encode(contents).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="google/gemini-3-flash-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type};base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract all text visible in this image. "
                                "Return only the extracted text, nothing else."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        extracted = response.choices[0].message.content
        return {"message": extracted, "filename": file.filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
