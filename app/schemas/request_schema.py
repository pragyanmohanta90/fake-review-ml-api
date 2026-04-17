from pydantic import BaseModel, Field
from datetime import datetime


class ReviewRequest(BaseModel):
    # =========================
    # NLP INPUT
    # =========================
    review_text: str = Field(..., min_length=3)

    # =========================
    # BASIC INFO
    # =========================
    rating: float = Field(..., ge=0, le=5)

    # =========================
    # BEHAVIOR FEATURES
    # =========================
    reviewUsefulCount: int = 0
    usefulCount: int = 0
    coolCount: int = 0
    funnyCount: int = 0
    reviewCount: int = 0
    friendCount: int = 0

    # =========================
    # TIME FEATURES
    # =========================
    order_time: datetime
    review_time: datetime