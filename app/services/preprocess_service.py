import pickle
import numpy as np

# load vectorizer
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def prepare_features(text):
    # TEXT FEATURES
    X_text = vectorizer.transform([text]).toarray()

    # BEHAVIOR FEATURES (dummy values)
    rating = 5
    reviewUsefulCount = 0
    usefulCount = 0
    coolCount = 0
    funnyCount = 0
    reviewCount = 1
    friendCount = 0
    ReviewLength = len(text)

    behavior = np.array([[
        rating,
        reviewUsefulCount,
        usefulCount,
        coolCount,
        funnyCount,
        reviewCount,
        friendCount,
        ReviewLength
    ]])

    # COMBINE
    X = np.hstack([X_text, behavior])

    return X