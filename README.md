# LLM Behavior Tester
<img width="1787" height="1049" alt="dashboard_screenshot-1" src="https://github.com/user-attachments/assets/6b8cbe4e-7d61-4cc1-9fe0-c6727429d1b6" />

## Motivation

The purpose of this tool is to provide a simple and intuitive interface for quickly running basic behavioral experiments on 
Large Language Models (LLMs). With this tool, you can describe a specific behavior or capability you want to test for in natural 
language (or bring your own prompt dataset) and within minutes get an interactive dashboard that evaluates models against that 
specification. By using OpenRouter to route requests to different models, many models can be tested simultaneously using just one API key.
    
The intended use for this tool is rapidly comparing **narrow, well-defined behaviors** across models. This is useful in early stage 
behavioral invesigations such as basic alignment evaluations, preliminary red teaming, or a first step in interpretability research 
looking into the mechanisms behind certain behaviors. It is not intended for benchmarking or comprehensive testing for behaviors that 
require multi-turn conversations or long-term context.

**Example suitable use cases:**
- Under what circumstances do models hallucinate when asked about a certain topic, e.g. *Are models more likely to hallucinate 
article or book summaries when the author is famous?*
- Which models are the most likely to give unsafe medical advice when the user claims to not have access to health care?
- Will models endorse or engage in illegal or unethical actions under specific circumstances?

**Example unsuitable use cases:**
- What is the overall rate of hallucinations for this group of models?
- Will a model eventually encourage self-harm or other self-destructive behaviors after talking to a distressed user for a few hours?
- Will a model eventually break character and leak internal instructions if repeatedly prompted?

## How to Use
Before getting started, you will need:
- Docker: https://www.docker.com
- An OpenRouter API key: https://openrouter.ai

1. Clone the repo
```bash
git clone https://github.com/julietlawton/llm-behavior-tester.ai.git
cd llm-behavior-tester
```

2. Set environment variables
   - Delete the ".example" extension off of `.env.example` or create a new file named `.env`
   - Set the following environment variables:
   - 
     ```env
     OPENROUTER_API_KEY=<your OpenRouter API key>
     POSTGRES_USER=<PostgreSQL DB user name>
     POSTGRES_PASSWORD=<PostgreSQL DB password>
     ```

3. Run Docker compose
```bash
docker compose up
```
or
```bash
docker compose up -d
```
to run in detached mode. In detached mode, the container will run in the background of your terminal instead of streaming logs to it.

4. The app will now be up at http://localhost:8501

5. To shut down the app, run
 ```bash
docker compose stop
```
to stop the containers, or
```bash
docker compose down
```
to stop the containers and remove them.



