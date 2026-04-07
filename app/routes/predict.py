from fastapi import APIRouter
from app.schemas.request_schema import ReviewRequest
from app.services.model_service import ModelService
from app.services.preprocess_service import prepare_features

router = APIRouter()
model_service = ModelService()

@router.post("/predict")
def predict_review(data: ReviewRequest):
    try:
        text = data.review.lower()

        # =========================
        # RULE 1: TOO SHORT
        # =========================
        if len(text.split()) < 3:
            return {
                "prediction": "Fake",
                "fake_score": 0.7,
                "reason": "Too short review"
            }

        # =========================
        # RULE 2: SUSPICIOUS PATTERNS
        # =========================
        suspicious_patterns = [

            # 🔥 Over-hype / fake positive
            "best ever", "must buy", "must purchase",
            "everyone should buy", "everyone must buy",
            "all should buy", "all must buy",
            "everyone needs this", "you need this",
            "buy this right now", "don't miss this",
            "10/10", "100%", "perfect product",
            "life changing", "worth every penny",
            "you won't regret", "highly recommend to everyone",
            "best thing i have ever bought",
            "this changed my life",

            # 🔥 Repetition / spam
            "very very good", "so so good",
            "amazing amazing", "good good",
            "best best", "nice nice",
            "super super", "too too good",

            # 🔥 Marketing / promotion
            "limited time offer", "hurry up",
            "best seller", "top product",
            "guaranteed results", "exclusive deal",
            "offer ends soon", "click now",

            # 🔥 Emotional exaggeration
            "i am so happy", "so so happy",
            "extremely satisfied", "super happy",
            "i love this so much",
            "i can't believe how good this is",

            # 🔥 Forced influence language
            "everyone should buy", "no one should miss",
            "everyone must try", "you must try this",
            "you have to buy this",
            "everyone should have this",

            # 🔥 Too negative (fake hate)
            "worst ever", "do not buy",
            "never buy this", "avoid this product",
            "waste of money", "totally useless",
            "very bad product", "fake product",
            "scam", "fraud product",
            "completely useless", "not worth it",

            # 🔥 Extreme emotional hate
            "i hate this so much",
            "this ruined everything",
            "biggest mistake ever",
            "worst purchase of my life"
        ]

        if any(pattern in text for pattern in suspicious_patterns):
            return {
                "prediction": "Fake",
                "fake_score": 0.85,
                "reason": "Suspicious or promotional language"
            }

        # =========================
        # RULE 3: TOO MANY EXCLAMATIONS 🚩
        # =========================
        if text.count("!") >= 3:
            return {
                "prediction": "Fake",
                "fake_score": 0.8,
                "reason": "Excessive excitement (spam-like)"
            }

        # =========================
        # MODEL PREDICTION
        # =========================
        features = prepare_features(text)
        prediction, score = model_service.predict(features)

        return {
            "prediction": "Fake" if prediction == 1 else "Genuine",
            "fake_score": float(score)
        }

    except Exception as e:
        return {"error": str(e)}