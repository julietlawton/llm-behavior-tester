import os
import streamlit as st
import pandas as pd
import requests
import json
import time
import yaml

BASE_URL = os.getenv("BASE_URL", "http://api:8000")
UNUSED_PARAMS = ["tool_choice", "tools", "include_reasoning", "logprobs", "top_logprobs"]

def check_key():
    """Verify the OpenRouter API key is valid."""
    key = requests.get(url=f"{BASE_URL}/key")
    if key.ok:
        return True, ""
    if key.status_code == 401:
        return False, "Invalid OpenRouter API key. Check your environment variables."
    else:
        try:
            err = key.json()
        except ValueError:
            err = key.text
        return False, f"OpenRouter API key check failed ({key.status_code}): {err}"

@st.cache_data
def get_model_list():
    """Get available models from openrouter and cache them."""
    res = requests.get(url=f"{BASE_URL}/models")
    models = json.loads(res.content)
    models_df = pd.json_normalize(models["data"])
    models_df = models_df[models_df["id"] != "openrouter/auto"]
    
    # Strip unsupported parameters from the supported parameters for each model
    models_df["supported_parameters"] = models_df["supported_parameters"].apply(
        lambda params: [p for p in params if p not in UNUSED_PARAMS]
    )
    return models_df
    
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Load supported parameter configurations
with open("supported_params.yml", "r") as f:
    supported_params = yaml.load(f, Loader=yaml.SafeLoader)

# Initialize session state variables
session_state_defaults = {
    "valid_key": False,
    "saved_model_ids": [],
    "experiment_data": None,
    "generated_prompts": None,
    "prompt_generation_running": False,
    "model_configs": {},
    "experiment_id": "",
    "polling": False,
    "experiment_status": None,
    "experiment_completed": False
    
}
for key, value in session_state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("LLM Behavior Tester")
# Validate OpenRouter API key
if "valid_key" not in st.session_state:
    key_ok, msg = check_key()
    if not key_ok:
        st.error(msg)
        st.stop()

    st.session_state.valid_key = True

st.subheader("Step 1. Select the models you want to test")
models_df = get_model_list()

# Filter available model to just free to use models
only_free = st.toggle("Only show free models", value=False, help="Filter available models to only show free model variants")
if only_free:
    st.info(
        "Free models have some restrictions:\n\n"
        "- Free usage limits: If you’re using a free model variant (ID ending in :free), you can make up to 20 requests per minute.\n\n"
        "- Daily limits:\n"
        "  - If you have purchased less than 10 credits, you’re limited to 50 :free model requests per day.\n"
        "  - If you purchase at least 10 credits, your daily limit is increased to 1000 :free model requests per day.\n"
    )
display_df = models_df[models_df["id"].str.contains(":free")] if only_free else models_df

# Filter out irrelevant columns from the display df
display_df = display_df.drop(columns=[
    "canonical_slug", 
    "hugging_face_id", 
    "created", 
    "per_request_limits",
    "architecture.modality",
    "architecture.tokenizer",
    "architecture.input_modalities",
    "architecture.output_modalities",
    "architecture.instruct_type",
    "pricing.image",
    "pricing.audio",
    "pricing.web_search"
])
display_models = st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
)

# Keep the model picker and session state variable synced
if "model_picker" not in st.session_state:
    st.session_state.model_picker = st.session_state.saved_model_ids.copy()

def _commit_selected_models():
    st.session_state.saved_model_ids = st.session_state.model_picker

# Model picker widget
st.multiselect(
    "Select models",
    options=models_df["id"].tolist(),
    key="model_picker",
    on_change=_commit_selected_models
)
selected_ids = st.session_state.saved_model_ids

# Prompt dataset creation section
if st.session_state.saved_model_ids:
    st.subheader("Step 2. Create dataset")
    upload_dataset_tab, generate_dataset_tab = st.tabs(["Upload Dataset", "Generate Dataset"])
    
    # Dataset upload
    with upload_dataset_tab:
        uploaded_dataset = st.file_uploader(
            "Choose a CSV or JSON file:",
            type=["csv", "json"],
            accept_multiple_files=False
        )
        if uploaded_dataset is not None:
            if uploaded_dataset.type == "text/csv":
                uploaded_df = pd.read_csv(uploaded_dataset)
            else:
                uploaded_df = pd.read_json(uploaded_dataset)

            # Validate the data format
            if "user" not in uploaded_df.columns:
                st.error("Invalid data format: required column 'user' missing.")
            elif not set(uploaded_df.columns).issubset({"user", "system", "target"}):
                extra = set(uploaded_df.columns) - {"user", "system", "target"}
                st.error(f"Invalid data format: unsupported columns {extra}.")
            else:
                user_df = st.data_editor(uploaded_df, num_rows="dynamic")
                st.write(f"**Total items:** {len(user_df)}")

                if st.button("Use this dataset", key="use_uploaded_data_button"):
                    # Check for any missing or invalid values before committing
                    if user_df.isna().any().any() or (user_df.map(lambda v: isinstance(v, str) and v.strip() == "").any().any()):
                        st.error("Invalid data: dataset has empty or NaN values.")
                    elif not user_df.map(lambda v: isinstance(v, str)).all().all():
                        st.error("Invalid data: dataset has non-string values.")
                    else:
                        # Set the uploaded data as the data for the experiment
                        st.session_state.experiment_data = user_df
        else:
            st.caption("Allowed columns: 'user' (user prompt, required), 'system' (system prompt, optional), 'target' (expected answer, optional).")
      
    # Dataset generation      
    with generate_dataset_tab:
        # Filter out models that don't support structured outputs
        default_model = "meta-llama/llama-3.3-8b-instruct:free"
        all_models = models_df[models_df["id"] != default_model]
        all_models = all_models["id"][
            all_models["supported_parameters"].apply(
                lambda params: isinstance(params, list) and "structured_outputs" in params
            )
        ].values

        model_options = [default_model] + list(all_models)
        num_prompts = st.number_input("Number of prompts to generate:", value=10, min_value=1, max_value=200)
        model_selection = st.selectbox(
            "Prompt generation model:",
            model_options,
        )
        
        # Optional system prompt section
        system_prompt_toggle = st.toggle("Add system prompt", value=False, help="Add a system prompt to be paired with every user prompt.")
        st.caption(f"[Learn more about system prompts]({"https://blog.promptlayer.com/system-prompt-vs-user-prompt-a-comprehensive-guide-for-ai-prompts/"})")
        if system_prompt_toggle:
            system_prompt = st.text_area(
                "Add a system prompt:",
                placeholder="You are a helpful assistant...",
                help="This system prompt will be included with every user prompt.",
                height = 150
            )
    
        dataset_gen_prompt = st.text_area(
            "Describe the experiment you want to run:",
            placeholder="I want to test...",
            help="""
                Describe what you capacity or capability you are testing for. 
                The prompt generation model will use this information to generate prompts 
                that test for this behavior.""",
            height=150
        )
        
        generated_clicked = st.button(
            "Generate prompts", 
            icon=":material/wand_stars:",
            disabled=st.session_state.prompt_generation_running
        )

        if generated_clicked:
            if not dataset_gen_prompt.strip():
                # Temporarily display a warning message if prompt is empty
                warning_placeholder = st.empty()
                with warning_placeholder:
                    st.warning("Please enter an experiment description first.")
                time.sleep(3)
                warning_placeholder.empty()
            else:
                st.session_state.prompt_generation_running = True
                st.rerun()
        
        # Once button is clicked, send prompt generation request
        if st.session_state.prompt_generation_running:
            with st.spinner("Generating prompts...", show_time=True):
                prompt_req_body = {
                    "model_id": model_selection,
                    "experiment_description": dataset_gen_prompt
                }
                if system_prompt_toggle and system_prompt:
                    prompt_req_body["system_prompt"] = system_prompt
                prompt_res = requests.post(
                    url=f"{BASE_URL}/generate/prompts", 
                    json=prompt_req_body,
                    params={
                        "n": num_prompts
                    }
                )
            try:
                prompt_res.raise_for_status()
                data = prompt_res.json()
                
                # Save generated data and job metadata
                st.session_state.generated_prompts = {
                    "data" : pd.DataFrame(data["items"]),
                    "metadata": {
                        "total": data.get("total"),
                        "usage": data.get("usage"),
                        "errors": data.get("errors", []),
                    }
                }     
            except requests.HTTPError as e:
                st.error(f"Request failed: {prompt_res.status_code} - {prompt_res.text}")
            finally:
                # Mark job as finished and rerun to re-enable prompt generation button
                st.session_state.prompt_generation_running = False
                st.rerun()              
                    
        if st.session_state.generated_prompts is not None:
            # Open data in a df editor to let the user make changes, and save it in a separate session state 
            # variable to persist edits
            generated_df_edited = st.data_editor(st.session_state.generated_prompts["data"], num_rows="dynamic")
            st.session_state.generated_df = generated_df_edited
            
            # Show job metadata
            job_metadata = st.session_state.generated_prompts["metadata"]
            job_errors = job_metadata["errors"]
            n_errors = len(job_errors)
            if n_errors:
                st.badge(
                    f"{n_errors} job{"" if n_errors == 1 else "s"} failed", 
                        icon=":material/warning:", 
                        color="orange"
                )
                for e in job_errors:
                    st.warning(e["message"])
            else:
                st.badge("All jobs finished successfully", icon=":material/check:", color="green")
            st.write(f"**Total items generated:** {job_metadata["total"]} **Job cost:** ${job_metadata["usage"]}")
            
            # Set generated data as the experiment data
            if st.button("Use this dataset", key="use_generated_data_button"):
                st.session_state.experiment_data = generated_df_edited

# Experiment configuration section    
if st.session_state.saved_model_ids and st.session_state.experiment_data is not None:
    st.subheader("Step 3. Set Up Experiment")
    st.markdown("**Configure Models:** (Optional)")
    
    # Keep model configs in sync with saved model ids
    st.session_state.model_configs = {
        id: cfg for id, cfg in st.session_state.model_configs.items() if id in st.session_state.saved_model_ids
    }

    # For every selected model, create an empty configuration
    for id in st.session_state.saved_model_ids:
        st.session_state.model_configs.setdefault(id, {})

    selected_models = models_df[models_df["id"].isin(st.session_state.saved_model_ids)]
    # Create a configuration menu for each selected model
    for id in selected_models["id"].values:
        saved_cfg = st.session_state.model_configs.get(id, {})
        with st.expander(id, expanded=False):
            # Filter parameters down to ones that are supported both by the model and the app
            param_options = [
                p for p in selected_models["supported_parameters"][selected_models["id"] == id].values[0] 
                if p in supported_params
            ]
            # Display the appropriate widget for each type of parameter
            for param in param_options:
                param_info = supported_params[param]
                # Restore previously saved parameter values, if any
                init = saved_cfg.get(param, param_info.get("default", None))
                if param_info["type"] == "float":
                    st.slider(
                        param, 
                        key=f"{id}_{param}_slider",
                        min_value=param_info["range"][0],
                        max_value=None if len(param_info["range"]) == 1 else param_info["range"][1],
                        value=init,
                        help=param_info["description"]
                    )
                    if "explainer_url" in param_info:
                        st.caption(f"Learn more about this parameter: [Watch]({param_info["explainer_url"]})")
                elif param_info["type"] == "int":
                    st.number_input(
                        param,
                        key=f"{id}_{param}_numinput",
                        min_value=param_info["range"][0],
                        max_value=None if len(param_info["range"]) == 1 else param_info["range"][1],
                        value=init,
                        help=param_info["description"]
                    )
                    if "explainer_url" in param_info:
                        st.caption(f"Learn more about this parameter: [Watch]({param_info["explainer_url"]})")
                elif param_info["type"] == "boolean":
                    st.checkbox(
                        param,
                        key=f"{id}_{param}_checkbox",
                        value=init,
                        help=param_info["description"]
                    )
                elif param_info["type"] == "enum":
                    st.segmented_control(
                        param,
                        key=f"{id}_{param}_segmented_control",
                        options=param_info["options"],
                        default=init,
                        selection_mode="single",
                        help=param_info["description"]
                    )
            # On save, create the configuration object
            save_settings = st.button("Save Settings", key=f"{id}_save_button")
            if save_settings:
                cfg = {}
                for param in param_options:
                    info = supported_params[param]
                    if info["type"] == "float":
                        wkey = f"{id}_{param}_slider"
                    elif info["type"] == "int":
                        wkey = f"{id}_{param}_numinput"
                    elif info["type"] == "boolean":
                        wkey = f"{id}_{param}_checkbox"
                    elif info["type"] == "enum":
                        wkey = f"{id}_{param}_segmented_control"
                    else:
                        continue

                    val = st.session_state.get(wkey, None)
                    default_val = info.get("default", None)
                    
                    # If the parameter has changed from the default, add it
                    if default_val != val:
                        
                        # Set seed value if selected
                        if param == "seed":
                            cfg[param] = 42
                        else:
                            cfg[param] = val
                # Save configuration for this model to session state
                st.session_state.model_configs[id] = cfg          

    st.divider()
    st.markdown("**Configure Evaluator:**")
    
    possible_labels = None
    default_eval_model = "anthropic/claude-3-haiku"
    all_eval_models = models_df["id"][models_df["id"] != default_eval_model]
    eval_model_options = [default_eval_model] + list(all_eval_models)
    eval_model_selection = st.selectbox(
        "Evaluator model:",
        eval_model_options,
    )
    
    # Set evaluator type based on whether this model can support structured outputs
    eval_model_params = models_df["supported_parameters"][models_df["id"] == eval_model_selection].values[0]
    eval_type = "structured" if "structured_outputs" in eval_model_params else "free"

    evaluator_prompt = st.text_area(
        "Describe the criteria for the evaluator:",
        placeholder="I am testing for...\n\n"
        "Label A: Choose this label when...\n\n"
        "Label B: Choose this label when..."
        ,
        help="""Provide details of how you want the evaluator to judge each model response. 
        The evaluator will classify every response as one of the provided classification labels or correct/incorrect 
        if targets were included in the dataset and provide a justification.""",
        height=250
    )
    
    # Set possible evaluation labels
    if "target" not in st.session_state.experiment_data.columns:
        labels = st.text_input(
            "Enter classification labels for evaluator (comma-separated):", 
            help="""
            The possible classification categories for the evaluator. List the classification labels 
            separated by commas, e.g. "low,medium,high". Spaces will be interpreted as part of the 
            labels,e.g. "too high, too low" will become ["too high", "too low"]."""
        )
        if labels:
            possible_labels = [t.strip().lower() for t in labels.split(",")]
            st.markdown(f"**Evaluator classification labels:** {possible_labels}")
    # If the dataset has targets, the labels will just be correct/incorrect
    else:
        possible_labels = ["correct", "incorrect"]
        st.caption("""
            If the dataset you added in Step 2 includes a target column, 
            the evaluator will mark each response as correct/incorrect. For multiclass
            classification, remove the target column from the dataset."""
        )
    
    start_experiment = st.button(
        "Run Experiment", 
        icon=":material/experiment:",
        disabled=st.session_state.polling
    )
    
    # Launch experiment
    if start_experiment:
        cancel_experiment = st.button(
            "Cancel", 
            icon=":material/cancel:",
            disabled=not st.session_state.polling
        )
        payload = {
            "dataset": st.session_state.experiment_data.to_dict(orient="records"),
            "model_configs": st.session_state.model_configs,
            "evaluator_type": eval_type,
            "evaluator_config": {
                "evaluator_model_id": eval_model_selection,
                "evaluator_prompt": evaluator_prompt,
                "possible_labels": possible_labels,
            },
        }
        experiment_start_res = requests.post(f"{BASE_URL}/experiments/start", json=payload)
        try:
            # Once the ID for this experiment is received, start polling for status updates
            experiment_start_res.raise_for_status()
            st.session_state.experiment_id = experiment_start_res.json()["experiment_id"]
            st.session_state.polling = True
            st.rerun()
        except requests.HTTPError:
            st.error(f"Start failed: {experiment_start_res.status_code} - {experiment_start_res.text}")
    
    # If the experiment is running, show the cancel experiment button
    if st.session_state.polling and st.session_state.experiment_id:
        if st.button("Cancel", icon=":material/cancel:"):
            cancel_res = requests.post(
                f"{BASE_URL}/experiments/{st.session_state.experiment_id}/cancel"
            )
            if cancel_res.ok:
                st.session_state.experiment_completed = True
            else:
                st.error(f"Cancel failed: {cancel_res.status_code} - {cancel_res.text}")
    
    # Show experiment status tiles and progress bar
    if st.session_state.experiment_status:
        status = st.session_state.experiment_status
        total = int(status.get("total_jobs", 0) or 0)
        completed = int(status.get("completed", 0) or 0)
        progress = (completed / total) if total else 0.0
        st.progress(progress, text=f"{completed}/{total} completed")
        st.caption(f"Total usage (cost): ${status.get('total_usage', 0):.4f}")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Pending", status.get("pending", 0))
        c2.metric("Running", status.get("running", 0))
        c3.metric("Awaiting Eval", status.get("awaiting_eval", 0))
        c4.metric("Completed", status.get("completed", 0))
        c5.metric("Failed", status.get("failed", 0))
        c6.metric("Cancelled", status.get("cancelled", 0))

    # Poll for experiment status
    if st.session_state.polling and st.session_state.experiment_id is not None:
        placeholder = st.empty()
        status_res = requests.get(f"{BASE_URL}/experiments/{st.session_state.experiment_id}/status")
        if status_res.ok:
            status = status_res.json()
            st.session_state.experiment_status = status

            # If the experiment is done, stop polling
            if status.get("finished"):
                st.session_state.polling = False
                st.session_state.experiment_completed = True
                st.rerun()
            # If the experiment is not done, wait and then poll again
            else:
                time.sleep(2)
                st.rerun()  
        else:
            st.session_state.polling = False
            st.error(f"Retreiving experiment status failed: {status_res.status_code} - {status_res.text}")
    
    # If experiment finished, 
    if st.session_state.experiment_completed:
        st.success("Experiment finished.")
        st.page_link("pages/Experiment_Dashboard.py", label="Open Dashboard", icon=":material/bar_chart_4_bars:")
    