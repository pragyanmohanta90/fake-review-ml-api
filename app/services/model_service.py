import pickle
import numpy as np
from scipy.sparse import hstack
from datetime import datetime
import os

from app.services.preprocess_service import preprocess_input


class ModelService:
    def __init__(self):
        # =========================
        # PATH HANDLING (SAFE)
        # =========================
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, "models")

        # =========================
        # LOAD MODELS
        # =========================
        self.nlp_model = pickle.load(open(os.path.join(MODEL_PATH, "nlp_model.pkl"), "rb"))
        self.vectorizer = pickle.load(open(os.path.join(MODEL_PATH, "vectorizer.pkl"), "rb"))

        try:
            self.behavior_model = pickle.load(open(os.path.join(MODEL_PATH, "behavior_model.pkl"), "rb"))
        except Exception as e:
            print("Behavior model load failed:", e)
            self.behavior_model = None

    # =========================
    # NLP MODEL
    # =========================
    def compute_nlp_score(self, review_text):
        # 🔥 Clean text (IMPORTANT)
        review_text = preprocess_input(review_text)

        text_vec = self.vectorizer.transform([review_text])

        review_length = len(review_text)
        word_count = len(review_text.split())
        exclamation_count = review_text.count("!")
        question_count = review_text.count("?")

        extra = np.array([[ 
            review_length,
            word_count,
            exclamation_count,
            question_count
        ]])

        features = hstack([text_vec, extra])

        prob = self.nlp_model.predict_proba(features)[0][1]
        return prob

    # =========================
    # BEHAVIOR MODEL
    # =========================
    def compute_behavior_score(self, data):
        try:
            if self.behavior_model is None:
                return 0.3

            features = [[
                data.rating,
                data.reviewUsefulCount,
                data.usefulCount,
                data.coolCount,
                data.funnyCount,
                data.reviewCount,
                data.friendCount,
                len(data.review_text or "")
            ]]

            return self.behavior_model.predict_proba(features)[0][1]

        except Exception as e:
            print("Behavior error:", e)
            return 0.3

    # =========================
    # TIME GAP RULE 🔥
    # =========================
    def compute_time_score(self, data):
        try:
            order_time = data.order_time
            review_time = data.review_time

            # Convert string → datetime
            if isinstance(order_time, str):
                order_time = datetime.fromisoformat(order_time)
            if isinstance(review_time, str):
                review_time = datetime.fromisoformat(review_time)

            # Prevent negative gap
            gap = max(0, (review_time - order_time).total_seconds() / 60)

            # RULE-BASED LOGIC
            if gap < 5:
                return 0.9
            elif gap < 30:
                return 0.7
            elif gap < 60:
                return 0.5
            else:
                return 0.2

        except Exception as e:
            print("Time error:", e)
            return 0.3

    # =========================
    # FINAL HYBRID PREDICTION
    # =========================
    def predict(self, data):
        review_text = data.review_text or ""

        # ===== NLP =====
        nlp_score = self.compute_nlp_score(review_text)

        # ===== BEHAVIOR =====
        behavior_score = self.compute_behavior_score(data)

        # ===== TIME =====
        time_score = self.compute_time_score(data)

        # ===== FINAL SCORE =====
        final_score = (
            0.6 * nlp_score +
            0.25 * behavior_score +
            0.15 * time_score
        )

        # ===== DECISION =====
        if final_score > 0.75:
            status = "Fake"
        elif final_score > 0.5:
            status = "Suspicious"
        else:
            status = "Genuine"

        return {
    "fake_probability": float(round(final_score, 2)),
    "status": status,
    "scores": {
        "nlp": float(round(nlp_score, 2)),
        "behavior": float(round(behavior_score, 2)),
        "time": float(round(time_score, 2))
    }
}