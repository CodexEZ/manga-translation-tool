import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, TrOCRProcessor
from PIL import Image
import io
import base64

from ray import serve
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@serve.deployment(
 num_replicas=1,
  ray_actor_options={"num_cpus": 2, "num_gpus": 0.5}
)
@serve.ingress(app)
class MangaOCRDeployment:
  def __init__(self):
    model_path = 'model_weights'
    self.tokenizer = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base", cache_dir=model_path)
    self.model = AutoModelForImageTextToText.from_pretrained("kha-white/manga-ocr-base", cache_dir=model_path)
    self.processor = TrOCRProcessor.from_pretrained("kha-white/manga-ocr-base", cache_dir=model_path)

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.device)
  
  @app.post("/predict")
  async def predict(self, file:UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(self.device)

    with torch.no_grad():
      generate_ids = self.model.generate(pixel_values)
      generated_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    return {"text": generated_text}

manga_ocr_app = MangaOCRDeployment.bind()
