from dataclasses import dataclass
import json
from typing import List, Union
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from typing_extensions import Annotated
import requests
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = FastAPI()

class GenerationRequest(BaseModel):
    model_id: str
    experiment_description: str

@dataclass
class PromptWorker:
    name: str
    queue: asyncio.Queue[int]
    results: List[str]
    client: AsyncOpenAI
    request: GenerationRequest
    
    async def run(self):
        while True:
            n = await self.queue.get()
            print(f"worker {self.name} taking job")
            try:
                prompts = await generate_prompts(self.client, self.request, n)
                print(prompts)
                self.results.extend(prompts)
            finally:
                self.queue.task_done()
    

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

async def generate_prompts(
    client: AsyncOpenAI,
    request: GenerationRequest,
    n: int):
    
    # client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    response = await client.chat.completions.create(
        model=request.model_id,
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
            {"role": "user", "content": request.experiment_description},
        ],
        response_format=build_prompt_schema(n),
        
    )
    
    reply = response.choices[0].message.content
    try:
        items = json.loads(reply)["items"]
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON content")

    return [row for row in items]

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
async def fetch_prompts(
    request: GenerationRequest,
    n: int = Query(description="Number of prompts to generate", le=100)):
    
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    
    max_prompts_per_request = 10
    batches = [max_prompts_per_request] * (n // max_prompts_per_request)
    if n % max_prompts_per_request:
        batches.append(n % max_prompts_per_request)
        
    queue = asyncio.Queue()
    
    results = []
    for b in batches:
        print("Adding job to queue")
        await queue.put(b)
    
    max_workers = 10
    n_workers = min(len(batches), max_workers) 
    workers = [
        asyncio.create_task(
            PromptWorker(f"w{i}", queue, results, client, request).run()
        ) for i in range(n_workers)
    ]
    
    await queue.join()
    for w in workers:
        w.cancel()
    
    return JSONResponse(content={"items": results, "total": len(results)})

@app.post("/generate/")
def generate(input: str, max_completion_tokens: int | None = None):
    # client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    # response = client.chat.completions.create(model="anthropic/claude-opus-4.1",
    # messages=[
    #     {
    #         "role": "system",
    #         "content": "You are a talking dog",
    #     },
    #     {
    #     "role": "user",
    #     "content": f"{input}"
    #     }
    # ],
    # max_completion_tokens=max_completion_tokens,
    # usage={
    #     "include": True
    # }
    # )
    # print("Usage Stats:", response.json()['usage'])
    # # reply = response.choices[0].message.content
    # return response
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "anthropic/claude-3-opus",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "usage": {
            "include": True
        }
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response)
    print("Response:", response.json()['choices'][0]['message']['content'])
    print("Usage Stats:", response.json()['usage'])
    # return response