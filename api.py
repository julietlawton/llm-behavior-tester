import asyncio
import functools
import logging
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
    """Decorator for retrying failed web requests with exponential backoff.
    Retryable exceptions:
    - 408: Request timed out.
    - 429: Request was rate limited.
    - 500: Internal server error.
    - 502: Chosen model was down, or an invalid response was received.
    - 503: No model provider available for this request. 
    - 504: Gateway time out.

    Args:
        retries (int, Optional): Maximum number of retry attempts. Defaults to 3.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            initial_sleep = 0.5
            max_sleep = 10
            for attempt in range(1, retries+1):
                try:
                    return await func(*args, **kwargs)
                except HTTPException as e:
                    if e.status_code not in {408, 429, 500, 502, 503, 504} or attempt == retries:
                        raise
                    
                    sleep = min(initial_sleep * (2 ** attempt - 1), max_sleep)
                    sleep += random.uniform(0, sleep * 0.1)
                    await asyncio.sleep(sleep)
            return
        return wrapper
    return decorator

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Attach an httpx AsyncClient and DB connection pool to app.state and create background experiment 
    worker tasks on startup. Close on shutdown.
    """
    n_workers = 10
    max_pool_size = 20
    base_url = "https://openrouter.ai/api/v1"
    
    app.state.http = httpx.AsyncClient(base_url=base_url)
    logger.info(f"Initialized HTTP client for OpenRouter base_url={base_url}.")
    app.state.pool = await asyncpg.create_pool(dsn=DATABASE_URL, max_size=max_pool_size)
    logger.info(f"Created DB connection pool (max_size={max_pool_size}).")
    workers = [
        asyncio.create_task(ExperimentWorker(f"ExperimentWorker{i+1}").run()) for i in range(n_workers)
    ]
    try:
        yield
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        logger.info("Closing HTTP client.")
        await app.state.http.aclose()
        logger.info("Closing DB connection pool.")
        await app.state.pool.close()

app = FastAPI(lifespan=lifespan)
logger = logging.getLogger("uvicorn.error")

class PromptGenerationRequest(BaseModel):
    """Prompt generation request body.

    Attributes:
        model_id (str): OpenRouter model ID.
        experiment_description (str): User provided experiment description, used to instruct prompt generation for that experiment.
        system_prompt (str, Optional): User provided system prompt to be included with every user prompt.
    """    
    model_id: str
    experiment_description: str
    system_prompt: Optional[str] = None

class ExperimentDatasetRow(BaseModel):
    """Schema for a single row in an experiment dataset.

    Attributes:
        user (str): User prompt for generating a model response to be evaluated.
        system (str, Optional): Optional system prompt to be included with user prompt.
        target (str, Optional): Optional ground truth to compare the model response against.
    """ 
    user: str
    system: Optional[str] = None
    target: Optional[str] = None
    
    class Config:
        extra = "forbid"

class EvaluatorConfig(BaseModel):
    """Evaluator config model. Defines the evaluator to be used for each job in the experiment.

    Attributes:
        evaluator_model_id (str): OpenRouter model ID for the model to be used as the evaluator.
        evaluator_prompt (str): The user provided criteria for the evaluator.
        target (List[str], Optional): Optional ground truth to compare the model response against.
    """ 
    evaluator_model_id: str
    evaluator_prompt: str
    possible_labels: Optional[List[str]] = None
        
class ExperimentRequest(BaseModel):
    """Experiment creation request body.

    Args:
        dataset (List[ExperimentDatasetRow]): Dataset for this experiment. Each prompt/model pair will be processed as one job.
        model_configs (Dict[str, Dict[str, Any]]): Supported parameter configuration for each model to be tested.
        evaluator_type (Literal["structured", "free"]): Evaluator type. Can be either 'structured' or 'free'.
        evaluator_config (EvaluatorConfig): Configuration for the evaluator.
    """
    dataset: List[ExperimentDatasetRow]
    model_configs: Dict[str, Dict[str, Any]]
    evaluator_type: Literal["structured", "free"]
    evaluator_config: EvaluatorConfig
    
@dataclass
class ExperimentWorker:
    """Worker that processes experiment jobs in the background while the service is running.

    Attributes:
        name (str): Worker ID.
        min_sleep (float, Optional): Minimum amount of time this worker sleeps for (in seconds). Defaults to 0.1.
        max_sleep (float, Optional): Maximum amount of time this worker sleeps for (in seconds). Defaults to 3.0.
    """  
    name : str
    min_sleep : float = 0.1
    max_sleep: float = 3.0
    
    async def run(self) -> None:
        sleep = self.min_sleep
        logger.info(f"[{self.name}] Worker is ready.")
        while True:
            try:
                # Grab a job with pending work from the DB and mark it as running
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

                # If there are no jobs available, sleep
                if not row:
                    await asyncio.sleep(sleep)
                    sleep = min(self.max_sleep, sleep*2)
                else:
                    sleep = self.min_sleep
                    # If the job is marked as pending, generate a model response for the prompt
                    if row["prev_state"] == "pending":
                        logger.info(f"[{self.name}] Executing model response generation.")
                        try:
                            model_config = json.loads(row["model_config"])
                            model_response = await generate_response(
                                row["model_id"],
                                model_config,
                                row["user_prompt"],
                                row["system_prompt"]
                            )

                        # If the request failed, mark the job as failed
                        except Exception as e:
                            logger.error(
                                f"[{self.name}] Executing model response generation failed. "
                                f"{str(e)} "
                                f"View job error log for more details."
                            )
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
                        # If the response was generated successfully, add it to the job row and mark the
                        # status as awaiting eval
                        else:
                            logger.info(
                                f"[{self.name}] Executing model response generation successful. "
                                f"Updating job status to awaiting eval."
                            )
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

                    # If job is marked as awaiting eval, run evaluation on it
                    elif row["prev_state"] == "awaiting_eval":
                        logger.info(f"[{self.name}] Executing model response evaluation.")
                        
                        # Load the evaluator details
                        async with app.state.pool.acquire() as conn:
                            evaluator_details = await conn.fetchrow(
                                """
                                SELECT evaluator_type, evaluator_config
                                FROM experiments
                                WHERE id = $1
                                """, row["experiment_id"]
                            )
                            evaluator_type = evaluator_details["evaluator_type"]
                            evaluator_config = json.loads(evaluator_details["evaluator_config"])
                            evaluator_model_id = evaluator_config["evaluator_model_id"]
                            evaluator_prompt = evaluator_config["evaluator_prompt"]
                            possible_labels = evaluator_config["possible_labels"]
                        
                        # Get the model response for this job
                        async with app.state.pool.acquire() as conn:
                            model_output = await conn.fetchrow(
                                """
                                SELECT model_response
                                FROM jobs
                                WHERE id = $1
                                """, row["id"]
                            )
                            model_response = model_output["model_response"]
                        
                        # Send the model response to be evaluated    
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
                            logger.error(
                                f"[{self.name}] Executing model response evaluated failed. "
                                f"{str(e)} "
                                f"View job error log for more details."
                            )
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
                        # If evaluation was successful, mark job as completed
                        else:
                            logger.info(
                                f"[{self.name}] Executing model response evaluation successful. "
                                f"Updating job status to completed."
                            )
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
            # Raises when a worker task is cancelled
            except asyncio.CancelledError as e:
                logger.info(f"[{self.name}] Worker shutting down.")
                raise
            
            # Catch any DB related exceptions                            
            except Exception as e:
                logger.error(f"[{self.name}] {str(e)}")
                                    
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
    
    async def run(self) -> None:
        # While jobs are available, pull one from the queue and run it
        while True:
            n = await self.queue.get()
            logger.info(f"[{self.name}] Executing job n={n}.")
            try:
                result = await generate_prompts(self.request, n)
                for _, prompt in result["prompts"].items():
                    if self.request.system_prompt is not None:
                        self.results.append({"system": self.request.system_prompt,"user": prompt})
                    else:
                        self.results.append({"user": prompt})
                    
                self.usage.append(float(result["usage"]))
            except Exception as e:
                logger.error(f"[{self.name}] Worker exited with error: {repr(e)}")     
                # Parse error for additional metadata (present for provider errors and moderation errors)
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
                logger.info(f"[{self.name}] Worker done.")
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
    model_config: dict,
    user_prompt: str, 
    system_prompt: str = None) -> dict:
    """Generates a response from the specified model using the provided prompt.

    Args:
        model_id (str): OpenRouter model ID.
        model_config (dict): Supported parameter overrides for this model.
        user_prompt (str): The prompt to send to the model
        system_prompt (str, Optional): Optional system prompt to include with the user prompt.

    Returns:
        dict: Response with model response and usage cost of this request.
    """
    
    messages = (
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        if system_prompt
        else [{"role": "user", "content": user_prompt}]
    )
    
    payload = {
        "model": model_id,
        "messages": messages,
        "usage": {"include": True}
    }
    
    # If the user provided values for supported parameters, add them to the request
    for param, val in model_config.items():
        payload[param] = val
 
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
    target_response: str = None) -> dict:
    """Sends a model response to the evaluator model to get a classification (with justification) back.
    If the evaluator has type structured, the response is a JSON object with required fields 'label' and
    'justification'. If the evaluator has type free, it will be instructed to return a response with special
    tags that are parsed to get the label and justification.

    Args:
        evaluator_model_id (str): The OpenRouter model id for the evaluator model.
        evaluator_type (Literal['structured', 'free']): The evaluator type. Either 'structured' or 'free'.
        evaluator_prompt (str): The criteria to be used by the evaluator.
        possible_labels (List[str]): The possible labels the evaluator can classify a response as.
        model_response (str): The model response to evaluate.
        target_response (str, Optional): Optional ground truth response to compare model response against.


    Returns:
        dict: Response with evaluation label, justification, and usage cost of this request.
    """    

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
    
    # If the evaluator is structured, specify response format otherwise ask model to use special tags
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
        
        # For structured responses, parse JSON object for required fields
        if evaluator_type == "structured":
            eval_result = json.loads(content)
            label = eval_result["label"]
            justification = eval_result["justification"]
            if label not in possible_labels:
                raise ValueError(f"Label '{label}' not in allowed set {possible_labels}.")
        
        # For free responses, use regex to parse text response for special tags
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
        dict: Response with generated user prompts and usage cost of this request.
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
    """Create prompt generation job. Splits the requested number of prompts into batches, assigns them to
    workers, and aggregates results.

    Args:
        request (PromptGenerationRequest): Request object with the model ID and experiment description.
        n (int): Number of prompts to generate.

    Returns:
        dict: JSON response object with generated user prompts, total number of prompts created, 
        and usage cost of this request.
    """
    
    # Split requested prompts into batches
    max_prompts_per_request = 10
    batches = [max_prompts_per_request] * (n // max_prompts_per_request)
    if n % max_prompts_per_request:
        batches.append(n % max_prompts_per_request)
        
    results, usage, errors = [], [], []
    queue = asyncio.Queue()

    # Put each batch size into task queue
    for b in batches:
        await queue.put(b)
    
    max_workers = 10
    n_workers = min(len(batches), max_workers) 
    
    # Create workers that will pull tasks from the queue and execute
    workers = [
        asyncio.create_task(
            PromptWorker(f"prompt_worker_{i}", queue, results, usage, errors, request).run()
        ) for i in range(n_workers)
    ]
    
    # Wait for all tasks in the queue to be executed, then cancel workers
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
 
@app.post("/experiments/start")
async def start_experiment(request: ExperimentRequest) -> dict:
    """Start a new experiment.

    Args:
        request (ExperimentRequest): request body with experiment configuration.

    Returns:
        dict: Response with ID of the created experiment.
    """

    dataset_json = [row.model_dump(exclude_none=True) for row in request.dataset]
    models_json = request.model_configs
    evaluator_json = request.evaluator_config.model_dump(exclude_none=True)

    async with app.state.pool.acquire() as conn:
        try:
            # Put new experiment in the experiments table
            async with conn.transaction():
                experiment_id = await conn.fetchval(
                    """
                    INSERT INTO experiments (dataset, models, evaluator_type, evaluator_config)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    json.dumps(dataset_json),
                    json.dumps(models_json),
                    request.evaluator_type,
                    json.dumps(evaluator_json),
                )

                # Break experiment into jobs and add to jobs table
                await conn.executemany(
                    """
                    INSERT INTO jobs (
                        experiment_id, 
                        model_id, 
                        model_config, 
                        job_status, 
                        user_prompt, 
                        system_prompt, 
                        target_response
                    )
                    VALUES ($1, $2, $3, 'pending', $4, $5, $6)
                    """, [
                    (
                        experiment_id, model_id, 
                        json.dumps(model_config), 
                        row["user"], 
                        row.get("system"), 
                        row.get("target")
                    )
                    for row in dataset_json
                    for model_id, model_config in models_json.items()
                ])

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Experiment creation failed: {str(e)}")

    return {"experiment_id": experiment_id}

@app.get("experiments/{experiment_id}/status")
async def check_experiment_status(experiment_id: int):
    return