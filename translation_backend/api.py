import os
import io
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# 1. Global Initialization (Runs once when server starts)
# ---------------------------------------------------------
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = ""  # Replace securely

# Initialize LLM & Chain
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a professional manga translator. Translate the following text from {input_lang} to {output_lang}. RETURN ONLY TRANSLATED TEXT!"),
    ("human", "{text}"),
])
translation_chain = prompt_template | llm | StrOutputParser()

# Initialize YOLO
model = YOLO("model_weights/best.pt")

# Initialize Font
try:
    font = ImageFont.truetype("arial.ttf", 16) 
except IOError:
    print("Warning: arial.ttf not found. Using default font.")
    font = ImageFont.load_default()

# Initialize FastAPI app
app = FastAPI(title="Manga Translation API")

# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------
def get_ocr_text(image_crop: np.ndarray, ocr_url: str = "http://localhost:8000/predict") -> str:
    """Sends a cropped image array to the local OCR server and returns the text."""
    if image_crop.size == 0:
        return ""
        
    pil_img = Image.fromarray(image_crop)
    byte_buffer = io.BytesIO()
    pil_img.save(byte_buffer, format="JPEG")
    image_bytes = byte_buffer.getvalue()

    response = requests.post(ocr_url, files={'file': image_bytes})
    if response.status_code == 200:
        return response.json().get('text', '')
    
    print(f"OCR Error {response.status_code}: {response.text}")
    return ""

def translate_text(japanese_text: str) -> str:
    """Translates Japanese text to English using Gemini."""
    if not japanese_text.strip():
        return ""
        
    return translation_chain.invoke({
        "input_lang": "Japanese",
        "output_lang": "English",
        "text": japanese_text
    })

def draw_translated_text(draw: ImageDraw.Draw, box: tuple, text: str):
    """Erases the original text box and draws horizontally wrapped English text."""
    x1, y1, x2, y2 = box
    
    # 1. Erase original text
    draw.rectangle([x1, y1, x2, y2], fill="white")
    
    box_width = x2 - x1
    box_height = y2 - y1
    
    # 2. Wrap text
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        if draw.textlength(test_line, font=font) <= (box_width - 4):
            current_line = test_line
        else:
            if current_line: 
                lines.append(current_line.strip())
            current_line = word + " "
    if current_line: 
        lines.append(current_line.strip())

    # 3. Calculate vertical centering
    _, _, _, line_height = draw.textbbox((0, 0), "A", font=font)
    total_text_height = line_height * len(lines)
    current_y = y1 + (box_height - total_text_height) / 2 
    
    # 4. Draw text horizontally centered
    for line in lines:
        line_width = draw.textlength(line, font=font)
        current_x = x1 + (box_width - line_width) / 2
        draw.text((current_x, current_y), line, fill="black", font=font)
        current_y += line_height

def process_manga_page(image_bytes: bytes) -> io.BytesIO:
    """Core pipeline: Detect -> OCR -> Translate -> Draw."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    results = model.predict(source=img, show=False)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if model.names[class_id] == 'text':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop image
                text_crop = np.array(img)[y1:y2, x1:x2]
                
                # Step 1: OCR
                jp_text = get_ocr_text(text_crop)
                
                if jp_text:
                    print(f"OCR Detected: {jp_text}")
                    
                    # Step 2: Translate
                    en_text = translate_text(jp_text)
                    print(f"Translated to: {en_text}")
                    
                    # Step 3: Draw
                    draw_translated_text(draw, (x1, y1, x2, y2), en_text)

    # Save final image to a bytes buffer to return via API
    output_buffer = io.BytesIO()
    img.save(output_buffer, format="PNG")
    output_buffer.seek(0) # Reset buffer pointer to the beginning
    
    return output_buffer

# ---------------------------------------------------------
# 3. FastAPI Endpoint
# ---------------------------------------------------------
@app.post("/process-manga")
async def process_manga_endpoint(file: UploadFile = File(...)):
    """API endpoint to upload a raw manga page and get a translated page back."""
    # if not file.content_type.startswith("image/"):
    #     raise HTTPException(status_code=400, detail="File must be an image.")
        
    image_bytes = await file.read()
    
    try:
        processed_image_buffer = process_manga_page(image_bytes)
        # StreamingResponse sends the bytes back directly as an image file
        return StreamingResponse(processed_image_buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))