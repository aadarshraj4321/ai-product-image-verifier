import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_similarity(image, text):
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # image-text similarity
    probs = logits_per_image.softmax(dim=1)
    return probs[0][0].item()
