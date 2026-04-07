from fastapi import APIRouter
from app.schemas.request_schema import ReviewRequest
from app.services.model_service import ModelService
from app.services.preprocess_service import clean_text

router = APIRouter()
model_service = ModelService()

@router.post("/predict")
def predict_review(data: ReviewRequest):
    try:
        text = data.review

        #  RULE-BASED CHECK 
        if len(text.split()) < 3:
            return {
                "fake_score": 0.7,
                "label": "fake",
                "reason": "Too short review"
            }

        # NORMAL MODEL FLOW
        cleaned = clean_text(text)
        score = model_service.predict(cleaned)

        return {
            "fake_score": float(score),
            "label": "fake" if score > 0.5 else "real"
        }

    except Exception as e:
        return {"error": str(e)}