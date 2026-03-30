from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
import matplotlib.pyplot as plt
import io
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAtZfIx6G5VYu2NjN_ay09ljgy0BLp30RI"

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

# 2. Create a Translation Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional manga translator. Translate the following text from {input_lang} to {output_lang}. RETURN ONLY TRANSLATED TEXT!"),
    ("human", "{text}"),
])
model = YOLO("model_weights/best.pt")
chain = prompt | llm | StrOutputParser()
img = Image.open("manga_panel_jp.png").convert("RGB")
draw = ImageDraw.Draw(img)
results = model.predict(source=img, show=False)

try:
    font = ImageFont.truetype("arial.ttf", 16) 
except IOError:
    print("Warning: arial.ttf not found. Using default font.")
    font = ImageFont.load_default()

for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        if model.names[class_id] == 'text':
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            text_crop = np.array(img)[y1:y2,x1:x2]
            pil_img =  Image.fromarray(text_crop)
            byte_buffer = io.BytesIO()
            pil_img.save(byte_buffer, format = "JPEG")
            image_bytes = byte_buffer.getvalue()

            url = "http://localhost:8000/predict"
            files = {'file':image_bytes}
            response = requests.post(url,files = files)

            if response.status_code == 200:
                print(f"OCR Results : {response.json().get('text')}")
                translation = chain.invoke({
                    "input_lang":"Japanese",
                    "output_lang":"English",
                    "text":f"{response.json().get('text')}"
                })
                print(f"Translated : {translation}")
                draw.rectangle([x1, y1, x2, y2], fill="white")
                
                box_width = x2 - x1
                box_height = y2 - y1
                
                # 2. Wrap the translated text to fit the box width
                words = translation.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + word + " "
                    # Check if the line width exceeds the bounding box width (minus a small margin)
                    if draw.textlength(test_line, font=font) <= (box_width - 4):
                        current_line = test_line
                    else:
                        if current_line: 
                            lines.append(current_line.strip())
                        current_line = word + " "
                if current_line: 
                    lines.append(current_line.strip())

                # 3. Calculate text height to vertically center it
                # Get the height of a single line of text
                _, _, _, line_height = draw.textbbox((0, 0), "A", font=font)
                total_text_height = line_height * len(lines)
                
                # Starting Y position (vertically centered)
                current_y = y1 + (box_height - total_text_height) / 2 
                
                # 4. Draw each line, horizontally centered
                for line in lines:
                    line_width = draw.textlength(line, font=font)
                    current_x = x1 + (box_width - line_width) / 2
                    
                    # Draw the text in black
                    draw.text((current_x, current_y), line, fill="black", font=font)
                    current_y += line_height
            else:
                print(f"Error {response.status_code}: {response.text}")
            
img.save("manga_panel_translated.png")