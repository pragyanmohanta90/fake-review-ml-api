from fastapi import FastAPI
from app.routes.predict import router

app = FastAPI(title="Fake Review Detection API")

app.include_router(router)

@app.get("/")
def home():
    return {"message": "API is running"}