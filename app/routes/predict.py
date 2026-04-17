from fastapi import APIRouter
from app.schemas.request_schema import ReviewRequest
from app.services.model_service import ModelService

router = APIRouter()
model_service = ModelService()


@router.post("/predict")
def predict_review(data: ReviewRequest):
    try:
        text = data.review_text or ""
        text_lower = text.lower()

        # =========================
        # RULE 1: TOO SHORT
        # =========================
        if len(text_lower.split()) < 3:
            return {
                "fake_probability": 0.7,
                "status": "Fake",
                "scores": {
                    "nlp": 0,
                    "behavior": 0,
                    "time": 0
                },
                "reasons": ["Too short review"]
            }

        # =========================
        # RULE 2: SUSPICIOUS PATTERNS
        # =========================
        suspicious_patterns = [
            "best ever", "must buy", "everyone should buy",
            "10/10", "100%", "perfect product",
            "very very good", "amazing amazing",
            "limited time offer", "hurry up",
            "worst ever", "do not buy", "scam"
        ]

        if any(pattern in text_lower for pattern in suspicious_patterns):
            return {
                "fake_probability": 0.85,
                "status": "Fake",
                "scores": {
                    "nlp": 0,
                    "behavior": 0,
                    "time": 0
                },
                "reasons": ["Suspicious or promotional language"]
            }

        # =========================
        # RULE 3: TOO MANY EXCLAMATIONS
        # =========================
        if text_lower.count("!") >= 3:
            return {
                "fake_probability": 0.8,
                "status": "Fake",
                "scores": {
                    "nlp": 0,
                    "behavior": 0,
                    "time": 0
                },
                "reasons": ["Excessive excitement (spam-like)"]
            }

        # =========================
        # HYBRID MODEL CALL
        # =========================
        result = model_service.predict(data)

        return result

    except Exception as e:
        return {"error": str(e)}