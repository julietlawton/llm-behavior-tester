from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from typing_extensions import Annotated
import requests
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/models/")
def list_models():
    response = requests.get("https://openrouter.ai/api/v1/models")
    return response.json()