from fastapi import FastAPI, Request
from transformers import pipeline
import torch

app = FastAPI()

# ✅ Use a lightweight sentiment model that works well on CPU
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

device = 0 if torch.cuda.is_available() else -1
moderation_pipeline = pipeline("text-classification", model=MODEL_NAME, device=device)

@app.get("/")
def home():
    return {"message": "Debate Moderation API is live!"}

@app.post("/moderate/")
async def moderate_text(request: Request):
    data = await request.json()
    text = data.get("text", "")

    if not text:
        return {"error": "No text provided"}

    try:
        result = moderation_pipeline(text[:500])  # limit input length
        label = result[0]["label"].lower()

        if "toxic" in label or "negative" in label:
            action = "warn"
        elif "neutral" in label or "positive" in label:
            action = "allow"
        else:
            action = "allow"

        return {"text": text, "action": action, "label": label}
    except Exception as e:
        return {"error": str(e)}
