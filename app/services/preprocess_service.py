# app/services/preprocess.py

import re


# =========================
# TEXT CLEANING FUNCTION
# =========================
def clean_text(text: str) -> str:
    """
    Clean and normalize review text
    """
    text = str(text).lower()

    # remove URLs
    text = re.sub(r"http\S+", "", text)

    # remove special characters (keep ! ?)
    text = re.sub(r"[^a-zA-Z0-9!?]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# OPTIONAL: SAFE INPUT HANDLER
# =========================
def preprocess_input(text: str) -> str:
    """
    Wrapper for safe preprocessing
    (can extend later if needed)
    """
    if not text:
        return ""

    return clean_text(text)