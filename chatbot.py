from pathlib import Path
import json
import random
from .nlp import predict_intent
from .rules import rule_response

BASE = Path(__file__).resolve().parent.parent
DATA = json.loads((BASE / "data" / "intents.json").read_text(encoding="utf-8"))
RESPONSES = {item["tag"]: item["responses"] for item in DATA["intents"]}

CONFIDENCE_THRESHOLD = 0.55

def get_bot_response(text: str):
    text = text.strip()

    if not text:
        return {"intent": "vacio", "confidence": 0.0, "response": "Escribe una consulta para poder ayudarte."}

    intent, confidence = predict_intent(text)

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "intent": intent,
            "confidence": confidence,
            "response": (
                "No estoy completamente seguro de haber entendido. "
                "Puedes preguntarme por modelos, precios, crédito, economía, trabajo, ciudad o aventura."
            )
        }

    response = rule_response(intent, text)
    if not response:
        response = random.choice(RESPONSES.get(intent, ["¿Puedes reformular tu consulta?"]))

    return {"intent": intent, "confidence": confidence, "response": response}
