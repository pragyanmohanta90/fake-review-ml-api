import pickle
import numpy as np

class ModelService:
    def __init__(self):
        self.model = pickle.load(open("models/model.pkl", "rb"))
        self.vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

    def predict(self, text: str):
        # TEXT → TF-IDF
        vec = self.vectorizer.transform([text]).toarray()

        # EXTRA FEATURES (same as training)
        review_length = len(text)
        word_count = len(text.split())

        extra = np.array([[review_length, word_count]])

        # COMBINE
        final_input = np.hstack((vec, extra))

        # PREDICT
        prob = self.model.predict_proba(final_input)[0][1]

        return prob