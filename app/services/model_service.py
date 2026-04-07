import pickle

class ModelService:
    def __init__(self):
        self.model = pickle.load(open("models/final_model.pkl", "rb"))

    def predict(self, features):
        prediction = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0][1]  # fake probability

        return prediction, prob