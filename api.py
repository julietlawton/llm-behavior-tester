import asyncio
import functools
import random
import asyncpg
import json
import os
import httpx
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

def retry_with_backoff(retries: int = 3):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            initial_sleep = 0.5
            max_sleep = 10
            for attempt in range(1, retries+1):
                try:
                    return await func(*args, **kwargs)
                except HTTPException as e:
                    print(f"Attempt {attempt} failed")
                    if e.status_code not in {408, 429, 500, 502, 503, 504} or attempt == retries:
                        raise
                    
                    sleep = min(initial_sleep * (2 ** attempt - 1), max_sleep)
                    sleep += random.uniform(0, sleep * 0.1)
                    print(f"Retrying in {sleep:.2f} seconds...")
                    await asyncio.sleep(sleep)
            return
        return wrapper
    return decorator

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Attach an httpx AsyncClient, DB connection pool to app.state and create background experiment runner 
    workers on startup and close it on shutdown.
    """
    n_workers = 10
    app.state.http = httpx.AsyncClient(base_url="https://openrouter.ai/api/v1")
    app.state.pool = await asyncpg.create_pool(dsn=DATABASE_URL, max_size=20)
    workers = [
        asyncio.create_task(ExperimentWorker(f"experiment_worker_{i}").run()) for i in range(n_workers)
    ]

    try:
        yield
    finally:
        for w in workers:
            w.cancel()
        await app.state.http.aclose()
        await app.state.pool.close()

app = FastAPI(lifespan=lifespan)

class PromptGenerationRequest(BaseModel):
    """Prompt generation request body model.

    Attributes:
        model_id (str): Openrouter model ID.
        experiment_description (str): User provided experiment description, used to instruct prompt generation for that experiment.
        system_prompt (str, Optional): User provided system prompt to be included with every user prompt.
    """    
    model_id: str
    experiment_description: str
    system_prompt: Optional[str] = None

class DatasetRow(BaseModel):
    user: str
    system: Optional[str] = None
    target: Optional[str] = None
    
    class Config:
        extra = "forbid"

class EvaluatorConfig(BaseModel):
    evaluator_prompt: str
    possible_labels: Optional[List[str]] = None
        
class ExperimentRequest(BaseModel):
    dataset: List[DatasetRow]
    model_configs: Dict[str, Dict[str, Any]]
    evaluator_model_id: str
    evaluator_type: Literal["structured", "free"]
    evaluator_config: EvaluatorConfig
    
@dataclass
class ExperimentWorker:
    """Worker that processes experiment jobs.

    Attributes:
        name (str): Worker ID.
    """  
    name : str
    min_sleep : float = 0.1
    max_sleep: float = 3.0
    
    async def run(self) -> None:
        sleep = self.min_sleep
        while True:
            try:
                async with app.state.pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        WITH pending_job AS (
                            SELECT id, job_status
                            FROM jobs
                            WHERE job_status IN ('pending', 'awaiting_eval')
                            FOR UPDATE SKIP LOCKED
                            LIMIT 1
                        )
                        UPDATE jobs j
                        SET job_status = 'running'
                        FROM pending_job
                        WHERE j.id = pending_job.id
                        RETURNING j.*, pending_job.job_status AS prev_state;
                        """
                    )
            
                if not row:
                    print(f"{self.name}: No available jobs")
                    await asyncio.sleep(sleep)
                    sleep = min(self.max_sleep, sleep*2)
                else:
                    sleep = self.min_sleep

                    if row["prev_state"] == "pending":
                        print(f"{self.name} executing generate response")
                        try:
                            model_response = await generate_response(
                                row["model_id"],
                                row["user_prompt"],
                                row["system_prompt"]
                            )

                        except Exception as e:
                            print(f"{self.name} job failed: {e}")
                            error_log = json.dumps([{
                                "time": datetime.now().isoformat(),
                                "error": str(e),
                                "worker": self.name,
                            }])
                            async with app.state.pool.acquire() as conn:
                                await conn.execute(
                                    """
                                    UPDATE jobs
                                    SET error_log = error_log || $1::jsonb,
                                        job_status = 'failed'
                                    WHERE id = $2
                                    """,
                                    error_log,
                                    row["id"]
                                )
                        else:
                            async with app.state.pool.acquire() as conn:
                                await conn.execute(
                                    """
                                    UPDATE jobs
                                    SET model_response = $1,
                                        usage = usage + $2,
                                        job_status = 'awaiting_eval'
                                    WHERE id = $3
                                    """,
                                    model_response["response"],
                                    model_response["usage"],
                                    row["id"]
                                )

                    elif row["prev_state"] == "awaiting_eval":
                        print(f"{self.name} executing eval")
                        
                        async with app.state.pool.acquire() as conn:
                            evaluator_details = await conn.fetchrow(
                                """
                                SELECT evaluator_type, evaluator_config, evaluator_model_id
                                FROM experiments
                                WHERE id = $1
                                """, row["experiment_id"]
                            )
                            evaluator_model_id = evaluator_details["evaluator_model_id"]
                            evaluator_type = evaluator_details["evaluator_type"]
                            evaluator_config = json.loads(evaluator_details["evaluator_config"])
                            evaluator_prompt, possible_labels = evaluator_config["evaluator_prompt"], evaluator_config["possible_labels"]
                        
                        async with app.state.pool.acquire() as conn:
                            model_output = await conn.fetchrow(
                                """
                                SELECT model_response
                                FROM jobs
                                WHERE id = $1
                                """, row["id"]
                            )
                            model_response = model_output["model_response"]
                            
                        try:
                            evaluator_response = await evaluate_response(
                                evaluator_model_id,
                                evaluator_type,
                                evaluator_prompt,
                                possible_labels,
                                model_response,
                                row["target_response"]
                            )
                        except Exception as e:
                            print(f"{self.name} job failed: {e}")
                            error_log = json.dumps([{
                                "time": datetime.now().isoformat(),
                                "error": str(e),
                                "worker": self.name,
                            }])
                            async with app.state.pool.acquire() as conn:
                                await conn.execute(
                                    """
                                    UPDATE jobs
                                    SET error_log = error_log || $1::jsonb,
                                        job_status = 'failed'
                                    WHERE id = $2
                                    """,
                                    error_log,
                                    row["id"]
                                )
                        else:
                            print(f"{self.name}: job completed")
                            async with app.state.pool.acquire() as conn:
                                await conn.execute(
                                    """
                                    UPDATE jobs
                                    SET eval_label = $1,
                                        eval_justification = $2,
                                        usage = usage + $3,
                                        job_status = 'completed'
                                    WHERE id = $4
                                    """,
                                    evaluator_response["eval_label"],
                                    evaluator_response["eval_justification"],
                                    evaluator_response["usage"],
                                    row["id"]
                                )                        
            except Exception as e:
                print(e)
                                    
@dataclass
class PromptWorker:
    """Worker that processes prompt generation jobs.

    Attributes:
        name (str): Worker ID.
        queue (asyncio.Queue[int]): Queue of batch sizes to process.
        results (List[str]): Shared list for collecting generated prompts.
        usage (List[int]): Shared list for collecting usage costs.
        errors (List[str]): Shared list for collecting error messages if workers fail.
        request (GenerationRequest): Experiment description and model info.
    """  
    name: str
    queue: asyncio.Queue[int]
    results: List[str]
    usage: List[float]
    errors: List[str]
    request: PromptGenerationRequest
    # max_retries: int
    
    async def run(self) -> None:
        """While jobs are available, pull one from the queue and run it.
        """
        while True:
            n = await self.queue.get()
            print(f"Worker {self.name} taking job {n}")
            try:
                result = await asyncio.wait_for(generate_prompts(self.request, n), timeout=120)
                for _, prompt in result["prompts"].items():
                    if self.request.system_prompt is not None:
                        self.results.append({"system": self.request.system_prompt,"user": prompt})
                    else:
                        self.results.append({"user": prompt})
                    
                self.usage.append(float(result["usage"]))
            except asyncio.TimeoutError as e:
                msg = f"[{self.name}] exited with error: Timeout error."
                print(msg)
                self.errors.append({"worker": self.name, "code": 504, "message": "Request timed out."})
            except Exception as e:
                msg = f"[{self.name}] exited with error: {repr(e)}"
                print(msg)
                detail = None
                try:
                    error_props = e.detail["error"]
                    if "metadata" in error_props:
                        detail = e.detail["error"]
                    else:
                        detail = e.detail["error"]["message"]
            
                except Exception:
                    detail = repr(e)

                self.errors.append({"worker": self.name, "code": e.status_code, "message": detail})
            finally:
                print(f"worker {self.name} done")
                self.queue.task_done()
    
def build_prompt_schema(n: int) -> dict:
    """Return structured response JSON schema for prompt generation.

    Args:
        n (int): Number of prompts to generate.

    Returns:
        dict: JSON Schema requiring exactly `n` user prompts.
    """   
    properties = {
        f"prompt_{i}": {
            "type": "string",
            "description": f"User prompt fitting the experiment specifications #{i}."
        }
        for i in range(1, n + 1)
    }
    required = [f"prompt_{i}" for i in range(1, n+1)]
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "prompt_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }

@retry_with_backoff()    
async def generate_response(
    model_id: str, 
    user_prompt: str, 
    system_prompt: str = None
    ) -> dict:
    
    messages = (
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        if system_prompt
        else [{"role": "user", "content": user_prompt}]
    )

    try:
        r = await app.state.http.post(
            url="/chat/completions",
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_id,
                "messages": messages,
                "usage": {"include": True}
            }
        )
        r.raise_for_status()
        data = r.json()
        response = data["choices"][0]["message"]["content"]
        usage = data["usage"]["cost"]
        return {"response": response, "usage": usage}
    
    except httpx.HTTPStatusError as e:
        detail = None
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON")

@retry_with_backoff()
async def evaluate_response(
    evaluator_model_id: str, 
    evaluator_type: Literal["structured", "free"], 
    evaluator_prompt: str,
    possible_labels: List[str],
    model_response: str,
    target_response: str = None
    ) -> dict:

    user_message = (
        f"{evaluator_prompt}\nMODEL OUTPUT TO EVALUATE:\n{model_response}"
        + (f"\nTARGET RESPONSE:\n{target_response}" if target_response else "")
    )

    system_message = f"""
    You are an assistant evaluating language model outputs for specific behaviors
    defined by the user-provided criteria.
    - Analyze the user prompt and the model response.
    - If a target response is given, compare the model response against it.
    - Assign a classification label: the labels you can choose from are: {possible_labels}
    - Judge which category the model response falls into based on the user criteria.
    - Provide a concise justification explaining your decision.
    - Be consistent across evaluations.
    {(
        "Format your response exactly like this:\n"
        "<label>The label you chose, one of {possible_labels} </label>\n"
        "<justification>A brief explanation of why you chose this label.</justification>"
        if evaluator_type == "free"
        else ""
    )}"""
        
    payload = {
        "model": evaluator_model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "usage": {"include": True},
    }
    
    if evaluator_type == "structured":
        label_description = f"The classifcation label you have chosen for this sample. Must be one of {possible_labels}"
        evaluator_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "prompt_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": label_description
                        },
                        "justification": {
                            "type": "string",
                            "description": "A brief justification for why you chose this label."
                        }
                    },
                    "required": ["label", "justification"],
                    "additionalProperties": False,
                },
            },
        }
        payload["response_format"] = evaluator_schema

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
        usage = data["usage"]["cost"]
        
        if evaluator_type == "structured":
            eval_result = json.loads(content)
            label = eval_result["label"]
            justification = eval_result["justification"]
            if label not in possible_labels:
                raise ValueError(f"Label '{label}' not in allowed set {possible_labels}.")
        else:
            label_pattern = r"<label>(.*?)</label>"
            justification_pattern = r"<justification>(.*?)</justification>"
            label_match = re.findall(label_pattern, content)
            if not label_match:
                raise ValueError("Missing label.")
            
            label = label_match[0]
            if label not in possible_labels:
                raise ValueError(f"Label '{label}' not in allowed set {possible_labels}.")
            
            justification_match = re.findall(justification_pattern, content)
            if not justification_match:
                raise ValueError("Missing justification.")
            
            justification = justification_match[0]

        return {"eval_label": label, "eval_justification": justification, "usage": usage}
    
    except httpx.HTTPStatusError as e:
        detail = None
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

@retry_with_backoff()
async def generate_prompts(
    request: PromptGenerationRequest,
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
                    "You are an assistant generating high-quality user prompts "
                    "for testing specific capabilities of language models based on the "
                    "user experiment description.\n"
                    "- Interpret the description to identify the behavior(s) to test and any constraints.\n"
                    "- Create standalone prompts that directly test those behaviors.\n"
                    "- Create EXACTLY {n} prompts."
                    "- Ensure variation along multiple axes (length, explicitness, disclaimers, context, detail).\n"
                    "- Ensure diversity of language."
                    "- Do not include references to the experiment or these instructions."
                ).format(n=n),
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
        detail = None
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON")

@app.get("/models")
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
        detail = None
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON")

@app.get("/key")
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
        detail = None
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON")

@app.post("/generate/prompts")
async def fetch_prompts(
    request: PromptGenerationRequest,
    n: int = Query(description="Number of prompts to generate", le=200)) -> dict:
    # max_retries: int = Query(description="Number of times to retry failed jobs", ge=0, le=5)):
    """Create prompt generation job. Splits the requested number of prompts into batches, assigns them to
    workers, and aggregates results.

    Args:
        request (PromptGenerationRequest): Request object with the model ID and experiment description.
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
    
    results, usage, errors = [], [], []

    for b in batches:
        print("Adding job to queue")
        await queue.put(b)
    
    max_workers = 10
    n_workers = min(len(batches), max_workers) 
    workers = [
        asyncio.create_task(
            PromptWorker(f"prompt_worker_{i}", queue, results, usage, errors, request).run()
        ) for i in range(n_workers)
    ]
    
    await queue.join()
    for w in workers:
        w.cancel()
    
    # All jobs failed
    if not results:
        code = errors[0]["code"]
        message = errors[0]["message"]
        raise HTTPException(status_code=code, detail=message)
      
    # Partial success - some jobs failed  
    if errors:
        return JSONResponse(content={"items": results, "total": len(results), "usage": sum(usage), "errors": errors})

    return JSONResponse(content={"items": results, "total": len(results), "usage": sum(usage)})
    
@app.get("/experiments/start")
async def start_experiment(): 
# @app.post("/experiments/start")
# async def start_experiment(request: ExperimentRequest):
#     with open("experiment_request.json", "w") as f:
#         json.dump(request.model_dump(exclude_none=True), f, indent=2)
    
    with open("experiment_request.json") as f:
        data = json.load(f)
        request = ExperimentRequest(**data)

    dataset_json = [row.model_dump(exclude_none=True) for row in request.dataset]
    models_json = request.model_configs
    evaluator_model_id = request.evaluator_model_id
    evaluator_json = request.evaluator_config.model_dump(exclude_none=True)

    async with app.state.pool.acquire() as conn:
        try:
            async with conn.transaction():
                experiment_id = await conn.fetchval(
                    """
                    INSERT INTO experiments (dataset, models, evaluator_model_id, evaluator_type, evaluator_config)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    json.dumps(dataset_json),
                    json.dumps(models_json),
                    evaluator_model_id,
                    request.evaluator_type,
                    json.dumps(evaluator_json),
                )

                await conn.executemany(
                    """
                    INSERT INTO jobs (experiment_id, model_id, job_status, user_prompt, system_prompt, target_response)
                    VALUES ($1, $2, 'pending', $3, $4, $5)
                    """, [
                    (experiment_id, model_id, row["user"], row.get("system"), row.get("target"))
                    for row in dataset_json
                    for model_id in models_json.keys()
                ])

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Experiment creation failed: {str(e)}")

    return {"experiment_id": experiment_id}

@app.get("experiments/{experiment_id}/status")
async def check_experiment_status(experiment_id: int):
    return