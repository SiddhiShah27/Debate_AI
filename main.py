from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI()

# ==========================================================
# LIGHTWEIGHT MODEL (tiny or distilled variant)
# ==========================================================
# Using 'distilbert-base-uncased' instead of sentiment fine-tuned one
# since the latter consumes extra memory on Render Free Tier.

MODEL_NAME = "distilbert-base-uncased"

try:
    print("✅ Loading lightweight model...")
    classifier = pipeline("text-classification", model=MODEL_NAME, device=-1)
except Exception as e:
    print("⚠️ Model load failed:", e)
    print("Falling back to tiny model.")
    classifier = pipeline("text-classification", model="sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english", device=-1)

# ==========================================================
# Request schema
# ==========================================================
class Message(BaseModel):
    text: str


# ==========================================================
# Analysis logic
# ==========================================================
def analyze_message(text: str):
    if not text.strip():
        return {"label": "neutral", "score": 0.0, "action": "NONE"}

    try:
        result = classifier(text[:256])[0]  # shorter input
        label = result.get("label", "").lower()
        score = float(result.get("score", 0))

        if "toxic" in label or "negative" in label:
            action = "REMOVE"
        elif "warning" in label or "rude" in label:
            action = "WARN"
        else:
            action = "NONE"

        return {"label": label, "score": score, "action": action}
    except Exception as e:
        print("Error during analysis:", e)
        return {"label": "neutral", "score": 0.0, "action": "NONE"}


# ==========================================================
# Routes
# ==========================================================
@app.get("/")
def root():
    return {"status": "OK", "message": "DebateX AI Moderation API running ✅"}


@app.post("/analyze/")
def analyze_text(message: Message):
    result = analyze_message(message.text)
    return {"text": message.text, "result": result}


# ==========================================================
# Render Entrypoint (with dynamic port binding)
# ==========================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
