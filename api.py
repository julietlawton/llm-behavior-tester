import asyncio
import json
import os
import httpx
import requests
from dataclasses import dataclass
from typing import List, Union
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from contextlib import asynccontextmanager
from typing_extensions import Annotated

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Attach an httpx AsyncClient to app.state on startup and close it on shutdown.
    """
    app.state.http = httpx.AsyncClient(base_url="https://openrouter.ai/api/v1")
    try:
        yield
    finally:
        await app.state.http.aclose()

app = FastAPI(lifespan=lifespan)

class GenerationRequest(BaseModel):
    """Prompt generation request body model.

    Attributes:
        model_id (str): Openrouter model ID.
        experiment_description (str): User provided experiment description, used to instruct prompt generation for that experiment.
    """    
    model_id: str
    experiment_description: str

@dataclass
class PromptWorker:
    """Worker that processes prompt generation jobs.

    Attributes:
        name (str): Worker ID.
        queue (asyncio.Queue[int]): Queue of batch sizes to process.
        results (List[str]): Shared list for collecting generated prompts.
        usage (List[int]): Shared list for collecting usage costs.
        request (GenerationRequest): Experiment description and model info.
    """  
    name: str
    queue: asyncio.Queue[int]
    results: List[str]
    usage: List[float]
    request: GenerationRequest
    # max_retries: int
    
    async def run(self) -> None:
        """While jobs are available, pull one from the queue and run it.
        """
        while True:
            n = await self.queue.get()
            print(f"Worker {self.name} taking job {n}")
            try:
                result = await generate_prompts(self.request, n)
                self.results.extend(result["prompts"]["items"])
                self.usage.append(float(result["usage"]))
            except Exception as e:
                print(f"[{self.name}] error: {repr(e)}")
            finally:
                self.queue.task_done()

    
def build_prompt_schema(n: int) -> dict:
    """Return structured response JSON schema for prompt generation.

    Args:
        n (int): Number of prompts to generate.

    Returns:
        dict: JSON Schema requiring exactly `n` user prompts.
    """   
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
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    }

async def generate_prompts(
    request: GenerationRequest,
    n: int) -> dict:
    """Execute prompt generation request. 

    Args:
        request (GenerationRequest): Request object with the model ID and experiment description.
        n (int): Number of prompts to generate.

    Returns:
        dict: Dictionary with generated user prompts and usage cost of this request.
    """
    prompt_schema = build_prompt_schema(n)
    payload = {
        "model": request.model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant generating high-quality evaluation prompts "
                    "for testing specific capabilities of language models based on the "
                    "user experiment description.\n"
                    "- Interpret the description to identify the behavior(s) to test and any constraints.\n"
                    "- Create standalone prompts that directly test those behaviors.\n"
                    "- Ensure variation along multiple axes (length, explicitness, disclaimers, context, detail).\n"
                    "- Ensure diversity of language."
                ),
            },
            {"role": "user", "content": request.experiment_description},
        ],
        "response_format": prompt_schema,
        "usage": {"include": True},
    }

    try:
        r = await app.state.http.post(
            url="/chat/completions",
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        r.raise_for_status()

        data = r.json()
        content = data["choices"][0]["message"]["content"]
        prompts = json.loads(content)
        usage = data["usage"]["cost"]

        return {"prompts": prompts, "usage": usage}
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON")

@app.get("/models/")
async def list_models() -> dict:
    """List models available from OpenRouter.

    Returns:
        dict: JSON model list.
    """    
    try:
        r = await app.state.http.get("/models")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON")

@app.get("/check_key/")
async def check_key() -> dict:
    """Get information on the API key associated with the current authentication session.

    Returns:
        dict: API key information, including current usage and rate limits.
    """   
    try:
        r = await app.state.http.get(
            url="/key",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}"
            }
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON")

@app.post("/generate_prompts/")
async def fetch_prompts(
    request: GenerationRequest,
    n: int = Query(description="Number of prompts to generate", le=100)):
    # max_retries: int = Query(description="Number of times to retry failed jobs", ge=0, le=5)):
    """Create prompt generation job. Splits the requested number of prompts into batches, assigns them to
    workers, and aggregates results.

    Args:
        request (GenerationRequest): Request object with the model ID and experiment description.
        n (int): Number of prompts to generate.

    Returns:
        dict: JSON response object with generated user prompts, total number of prompts created, 
        and usage cost of this request.
    """
    max_prompts_per_request = 10
    batches = [max_prompts_per_request] * (n // max_prompts_per_request)
    if n % max_prompts_per_request:
        batches.append(n % max_prompts_per_request)
        
    queue = asyncio.Queue()
    
    results = []
    usage = []

    for b in batches:
        print("Adding job to queue")
        await queue.put(b)
    
    max_workers = 10
    n_workers = min(len(batches), max_workers) 
    workers = [
        asyncio.create_task(
            PromptWorker(f"w{i}", queue, results, usage, request).run()
        ) for i in range(n_workers)
    ]
    
    await queue.join()
    for w in workers:
        w.cancel()

    return JSONResponse(content={"items": results, "total": len(results), "usage": sum(usage)})

@app.post("/generate/")
async def generate(input: str, max_completion_tokens: int | None = None):
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
    url = "/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a talking dog",
            },
            {
            "role": "user",
            "content": f"{input}"
            }
        ],
        # Only one of "reasoning.effort" and "reasoning.max_tokens" can be specified
        # "reasoning": {
        #     "effort": "low",
        #     "max_tokens": 100,
        #     "exclude" : True
        # },
        "usage": {
            "include": True
        }
    }
    try:
        r = await app.state.http.post(url, headers=headers, json=payload)
        r.raise_for_status()
        # response = requests.post(url, headers=headers, data=json.dumps(payload))
        print(r)
        print(r.json())
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = data["usage"]["cost"]
        print("\nResponse:", content)
        print("\nUsage Stats:", usage)
        return JSONResponse(
            {
                "content": content,
                "usage": usage
            }
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON")
    