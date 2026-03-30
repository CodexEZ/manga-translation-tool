import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText,AutoProcessor,TrOCRProcessor
from PIL import Image
import os
import warnings
import logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 2. Silence the NumPy/SciPy version mismatch warning
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.sparse")

# 3. Silence specific Transformers weight-tying and processor warnings
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()



tokenizer = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base"
                                          ,cache_dir = r'D:\Projects\Manga Translator AI Model\model_weights')
model = AutoModelForImageTextToText.from_pretrained("kha-white/manga-ocr-base",
                                                    cache_dir=r'D:\Projects\Manga Translator AI Model\model_weights')
processor = TrOCRProcessor.from_pretrained("kha-white/manga-ocr-base",
                                           cache_dir=r'D:\Projects\Manga Translator AI Model\model_weights')
image = Image.open("image.png").convert("RGB")

pixel_values = processor(images = image, return_tensors = "pt").pixel_values
print(pixel_values.shape)

generate_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(generated_text)