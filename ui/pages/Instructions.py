import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.header("Motivation")
st.markdown(
"""
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
"""
)

st.header("Instructions")
st.subheader("Step 1: Choosing the models to test")
st.markdown(
"""
Select the models you want to test in the multi-select widget. You can view details about each model, such as pricing information 
and context length, in the models table.
You can also filter the table to show only models that are free to use. Requests made to these models are completely free, 
but they have strict rate limits and may be less suitable for experiments involving large numbers of prompts.
"""
)
st.image("images/models_screenshot.png")

st.subheader("Step 2: Uploading or generating a dataset")
st.markdown(
"""
Once you have selected at least one model, the dataset creation section will appear and you will be able to upload or generate a prompt dataset. 

**Uploading a dataset**

The file formats supported for uploading a dataset are JSON and CSV, and uploaded datasets must have the following schema: 

- **user** (text, required): User message to send to the model.
- **system** (text, optional): System prompt for each message that define how the model should behave.
- **target** (text, optional): The expected response from the model for a prompt.

You should include targets when you expect a specific response from the model. For example, if your dataset consists of multiple choice questions, 
the targets would be the correct answer for each question. If you include targets, model responses will be evaluated as correct/incorrect 
instead of using multi-label classification.

**Generating a dataset**

Alternatively, you can describe the experiment you want to run and have a prompt dataset generated for you. 
Select the number of prompts you want and in the text area, describe the behavior you are testing 
for and what kind of prompts the model should create. The more detailed you can be here, the higher quality 
the prompts will be. However, be careful when providing very specific examples as this can bias the model 
towards creating prompts that are very similar to the example, resulting in an overall lack of diversity. 
You can also include a system prompt, which will be paired with every user prompt. 

By default one of the free models is chosen for prompt generation, but you can choose a different model. 
In this step, only models that support structured outputs are available. Structured outputs is a generation mode 
that requires models to return a JSON formatted response matching a specific schema rather than a text response. 
This guarantees consistent outputs and allows prompts to be generated in batches, making the process cheaper and 
faster. For example, generating 25 prompts only requires 3 requests (10+10+5), instead of 25.

Click “**Generate Prompts**” to start the process. The screenshot below shows what you will see when prompt 
generation is complete. The dataset can be edited by double clicking on a cell to edit it. 
You can also add rows by clicking the plus icon at the bottom of the data editor or delete rows by selecting a 
row in the index column and pressing delete (or clicking the trash icon in the data editor tool tips). 
When you are finished, click "**Use this Dataset**" to move on to the next step.
"""
)
st.image("images/dataset_screenshot.png")

st.subheader("Step 3: Setting up the experiment")
st.markdown(
"""
After selecting the dataset you want to use, the final experiment configuration section will appear. 

**Configuring the models (Optional)**

For every model you selected in step 1, there will be a configuration widget for that model that allows you to optionally
override the default parameter values for that model, such as max_tokens and temperature. The supported parameters for each model 
can be seen in the model table in step 1.

**Configuring the evaluator**

The evaluator is the model that will be used to judge the test model responses. Select the model that you want to use as the evaluator, and then 
define the criteria it should use when judging model responses and the evaluator classification labels (if you are using a dataset with targets, 
you will not be able to enter classification labels and they will instead be correct/incorrect). Just like the dataset generation specification, 
the more detail you can provide here, the better. Try to provide very specific definitions of each label you are using. The evaluator will be
restricted to choosing one of the classification labels as the evaluation label for each response.

**Running the experiment**

Click **Run Experiment** to launch the experiment. Once the experiment has started, a progress bar and job status tiles
will appear for tracking the progress of the experiment. A cancel button will also appear that stops the experiment and
cancels any jobs that have not yet completed. For every experiment, two jobs are run for each prompt/model pair: one that sends
the prompt to the test model and gets a response, and one that sends that response to the evaluator model and gets an
evaluation label with a justification. You will be able to see the stage each job is in, as well as the running cost 
of the experiment (updated every few seconds).

Once the experiment has finished, you will be able to view the results in the experiment dashboard.
"""
)
st.image("images/eval_screenshot.png")

st.subheader("Step 4: Viewing the results of the experiment")
st.markdown(
"""
If you have evaluated more than one model, at the top of the dashboard there will be a grouped bar chart 
that compares results across models. This shows you how often each of the tested models exhibited each 
category of behavior. Beneath this there will be breakdowns for each model including per-model results and 
samples of responses that fell into each of the behavior categories. 

For example, in the experiment dashboard shown in the screenshot below, openai/gpt-5-mini refused to provide a summary for an
article it could not open for 88% of the tested prompts and hallucinated an article summary for 12% of the prompts. 
The card to the right of the chart shows a sample response for both categories, along with the prompt and evaluator 
justification. Toggling between response category examples will repopulate the card with a new sample.

At the bottom of the page you can view the full experiment data including request metadata and download it as a CSV file.
"""
)
st.image("images/dashboard_screenshot.png")