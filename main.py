from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import deque
import datetime
import logging
import json
import os

# ✅ Initialize FastAPI
app = FastAPI(title="Debate Moderation API (Fine-Tuned + Logging)")

# ✅ Enable CORS for Frontend Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Logging Configuration
logging.basicConfig(
    filename="moderation_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ✅ Load Fine-Tuned Model or Fallback
MODEL_PATH = "./debate_toxicity_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("✅ Loaded fine-tuned debate model")
except Exception as e:
    print(f"⚠️ Could not load fine-tuned model ({e}), using fallback model.")
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

moderation_model = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
)

# ✅ Rolling buffer (context)
context_buffer = deque(maxlen=5)

# ✅ File for structured logging
LOG_FILE = "live_moderation_log.json"

# ✅ Toxicity Classification Logic
def classify_toxicity(label: str, score: float):
    if "toxic" in label.lower() and score >= 0.85:
        return "high", "remove"
    elif "toxic" in label.lower() and score >= 0.6:
        return "medium", "mute"
    elif "toxic" in label.lower() and score >= 0.4:
        return "low", "warn"
    else:
        return "none", "no_action"

# ✅ Input Schema
class ModerationRequest(BaseModel):
    text: str
    user: str | None = None

# ✅ Main Moderation Endpoint
@app.post("/moderate")
def moderate_text(req: ModerationRequest):
    if not req.text.strip():
        return {"error": "Empty text"}

    entry = f"{req.user or 'unknown'}: {req.text}"
    context_buffer.append(entry)

    context_text = " ".join(context_buffer)
    results = moderation_model(context_text, truncation=True)
    label = results[0]["label"]
    score = float(results[0]["score"])
    level, action = classify_toxicity(label, score)

    log_msg = f"[User: {req.user}] [{label}] [Score: {score:.3f}] [Action: {action}] Text: {req.text}"
    logging.info(log_msg)
    print(log_msg)

    response = {
        "user": req.user,
        "text": req.text,
        "label": label,
        "score": round(score, 3),
        "toxicity_level": level,
        "action": action,
        "context_used": list(context_buffer),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return response

# ✅ New Endpoint: Log Events
@app.post("/log_action")
def log_action(data: dict):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user": data.get("user"),
        "text": data.get("text"),
        "action": data.get("action"),
        "score": data.get("score"),
        "toxicity_level": data.get("toxicity_level"),
    }

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([record], f, indent=2)
    else:
        with open(LOG_FILE, "r+") as f:
            logs = json.load(f)
            logs.append(record)
            f.seek(0)
            json.dump(logs, f, indent=2)

    return {"status": "logged", "recorded_action": record}

# ✅ Health Check
@app.get("/")
def health():
    return {"status": "API running", "model_loaded": MODEL_PATH}
