import json
from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI
from typing_extensions import Annotated
import requests
import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = FastAPI()

def build_prompt_schema(n: int = 5):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "promptgen",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": n,
                        "maxItems": n,
                        "items": {
                            "type": "object",
                            "properties": {
                                "user": {"type": "string", "minLength": 1},
                            },
                            "required": ["user"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": False
            }
        }
    }

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/models/")
def list_models():
    response = requests.get("https://openrouter.ai/api/v1/models")
    return response.json()

@app.get("/check_key/")
def check_key():
    response = requests.get(
        url="https://openrouter.ai/api/v1/key",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }
    )
    return response.json()

@app.post("/generate_prompts/")
def generate_prompts( 
    model_id: str, 
    experiment_description: str, 
    n: int):
   
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": (
                "You are an assistant generating high-quality evaluation prompts "
                "for testing specific capabilities of language models based on the "
                "user experiment description.\n"
                "- Interpret the description to identify the behavior(s) to test and any constraints.\n"
                "- Create standalone prompts that directly test those behaviors.\n"
                "- Ensure variation along multiple axes (length, explicitness, disclaimers, context, detail).\n"
                "- Ensure diversity of language."
            )},
            {"role": "user", "content": experiment_description},
        ],
        response_format=build_prompt_schema(n)

    )

    reply = response.choices[0].message.content
    try:
        content = json.loads(reply)
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON content")

    return JSONResponse(content=content)

@app.post("/generate/")
def generate(input: str, max_completion_tokens: int | None = None):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    response = client.chat.completions.create(model="openai/gpt-oss-20b:free",
    messages=[
        {
            "role": "system",
            "content": "You are a talking dog",
        },
        {
        "role": "user",
        "content": f"{input}"
        }
    ],
    max_completion_tokens=max_completion_tokens
    )

    reply = response.choices[0].message.content
    return reply