from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from collections import deque
import datetime

# Initialize FastAPI app
app = FastAPI(title="Debate Moderation API - Vercel Ready")

# Enable CORS (important for frontend integration)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing/demo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load lightweight model
# -------------------------------
# Using DistilBERT sentiment analysis as a base model
# (You can later switch to a fine-tuned debate model)
moderation_model = pipeline("sentiment-analysis")

# Rolling buffer for short-term transcript context
context_buffer = deque(maxlen=5)


# -------------------------------
# Request Schema
# -------------------------------
class ModerationRequest(BaseModel):
    text: str
    user: str | None = None


# -------------------------------
# Helper to classify toxicity
# -------------------------------
def classify_toxicity(text: str):
    result = moderation_model(text, truncation=True)[0]
    label = result["label"].lower()
    score = result["score"]

    # Simplified classification logic
    if label == "negative" and score > 0.8:
        level = "high"
        action = "remove"
    elif label == "negative" and score > 0.5:
        level = "medium"
        action = "mute"
    elif label == "negative":
        level = "low"
        action = "warn"
    else:
        level = "none"
        action = "none"

    return level, action, round(score, 3), label


# -------------------------------
# API Endpoint: Moderate Text
# -------------------------------
@app.post("/moderate")
def moderate_text(req: ModerationRequest):
    context_buffer.append(req.text)
    level, action, score, label = classify_toxicity(req.text)

    return {
        "user": req.user or "unknown",
        "text": req.text,
        "level": level,
        "action": action,
        "score": score,
        "label": label,
        "context": list(context_buffer),
        "timestamp": datetime.datetime.now().isoformat(),
    }


# -------------------------------
# Root Endpoint
# -------------------------------
@app.get("/")
def root():
    return {"message": "Debate Moderation API is live 🚀"}


# -------------------------------
# Vercel handler export
# -------------------------------
# Required for Vercel to detect the app
handler = app


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)

