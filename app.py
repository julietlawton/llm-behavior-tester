import streamlit as st
import pandas as pd
import requests
import json
import time
import yaml

BASE_URL = "http://127.0.0.1:8000"

def check_key():
    key = requests.get(url=f"{BASE_URL}/key/")
    if key.ok:
        return
    if key.status_code == 401:
        st.error("Invalid OpenRouter API key. Check your environment variables.")
    else:
        try:
            err = key.json()
        except ValueError:
            err = key.text
        st.error(f"OpenRouter API key check failed ({key.status_code}): {err}")
        
    
st.set_page_config(layout="wide")

with open("supported_params.yml", "r") as f:
    supported_params = yaml.load(f, Loader=yaml.SafeLoader)

UNUSED_PARAMS = ["tool_choice", "tools", "include_reasoning", "logprobs", "top_logprobs"]

st.title("LLM Behavior Tester")
# check_key()
st.subheader("Step 1. Select the models you want to test")

res = requests.get(url=f"{BASE_URL}/models/")
models = json.loads(res.content)
models_df = pd.json_normalize(models["data"])
models_df = models_df[models_df["id"] != "openrouter/auto"]
models_df["supported_parameters"] = models_df["supported_parameters"].apply(
    lambda params: [p for p in params if p not in UNUSED_PARAMS]
)

if "saved_model_ids" not in st.session_state:
    st.session_state.saved_model_ids = set()
    
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

model_picker = st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
)

selected_rows = model_picker.selection.rows
selected_models = models_df.iloc[selected_rows]
current_model_ids = selected_models["id"].tolist()
st.session_state.saved_model_ids = current_model_ids

if st.session_state.saved_model_ids:
    st.markdown("**Selected models:**")
    st.write(st.session_state.saved_model_ids)


if "experiment_data" not in st.session_state:
    st.session_state.experiment_data = None
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = None
if "prompt_generation_running" not in st.session_state:
    st.session_state.prompt_generation_running = False

if st.session_state.saved_model_ids:
    st.subheader("Step 2. Create dataset")
    upload_dataset_tab, generate_dataset_tab = st.tabs(["Upload Dataset", "Generate Dataset"])
    
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

            if "user" not in uploaded_df.columns:
                st.error("Invalid data format: required column 'user' missing.")
            elif not set(uploaded_df.columns).issubset({"user", "system", "target"}):
                extra = set(uploaded_df.columns) - {"user", "system", "target"}
                st.error(f"Invalid data format: unsupported columns {extra}.")
            else:
                user_df = st.data_editor(uploaded_df, num_rows="dynamic")
                st.write(f"**Total items:** {len(user_df)}")

                if st.button("Use this dataset", key="use_uploaded_data_button"):
                    if user_df.isna().any().any() or (user_df.map(lambda v: isinstance(v, str) and v.strip() == "").any().any()):
                        st.error("Invalid data: dataset has empty or NaN values")
                    elif not user_df.map(lambda v: isinstance(v, str)).all().all():
                        st.error("Invalid data: dataset has non-string values")
                    else:
                        st.session_state.experiment_data = user_df
        else:
            st.caption("Allowed columns: 'user' (user prompt, required), 'system' (system prompt, optional), 'target' (expected answer, optional).")
    with generate_dataset_tab:
        default_model = "openai/gpt-oss-20b:free"
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
                that test for this behavior. More detailed descriptions will generate better prompts.""",
            height=150
        )
        
        generated_clicked = st.button(
            "Generate prompts", 
            icon=":material/wand_stars:",
            disabled=st.session_state.prompt_generation_running
        )

        if generated_clicked:
            if not dataset_gen_prompt.strip():
                warning_placeholder = st.empty()
                with warning_placeholder:
                    st.warning("Please enter an experiment description first.")
                time.sleep(3)
                warning_placeholder.empty()
            else:
                st.session_state.prompt_generation_running = True
                st.rerun()
            
        if st.session_state.prompt_generation_running:
            with st.spinner("Generating prompts...", show_time=True):
                prompt_req_body = {
                    "model_id": model_selection,
                    "experiment_description": dataset_gen_prompt
                }
                if system_prompt_toggle and system_prompt:
                    prompt_req_body["system_prompt"] = system_prompt
                prompt_res = requests.post(
                    url=f"{BASE_URL}/generate/prompts/", 
                    json=prompt_req_body,
                    params={
                        "n": num_prompts
                    }
                )
            try:
                prompt_res.raise_for_status()
                data = prompt_res.json()
                st.session_state.generated_prompts = {
                    "data" : pd.DataFrame(data["items"]),
                    "metadata": {
                        "total": data.get("total"),
                        "usage": data.get("usage"),
                        "errors": data.get("errors", []),
                    }
                }
                    
            except requests.HTTPError as e:
                st.error(f"Request failed: {prompt_res.text}")  
                    
            finally:
                st.session_state.prompt_generation_running = False
                st.rerun()              
                    
        if st.session_state.generated_prompts is not None:
            generated_df_edited = st.data_editor(st.session_state.generated_prompts["data"], num_rows="dynamic")
            st.session_state.generated_df = generated_df_edited
            
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
            if st.button("Use this dataset", key="use_generated_data_button"):
                st.session_state.experiment_data = generated_df_edited


    
if st.session_state.experiment_data is not None:
    st.subheader("Step 3. Set Up Experiment")
    st.markdown("**Configure Models:** (Optional)")
    
    if "model_configs" not in st.session_state:
        st.session_state.model_configs = {id: {} for id in selected_models["id"].values}

    for id in selected_models["id"].values:
        with st.expander(id, expanded=False):
            param_options = [
                p for p in selected_models["supported_parameters"][selected_models["id"] == id].values[0] 
                if p in supported_params
            ]
            for param in param_options:
                param_info = supported_params[param]
                if param_info["type"] == "float":
                    st.slider(
                        param, 
                        key=f"{id}_{param}_slider",
                        min_value=param_info["range"][0],
                        max_value=None if len(param_info["range"]) == 1 else param_info["range"][1],
                        value=None if "default" not in param_info else param_info["default"],
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
                        value=None if "default" not in param_info else param_info["default"],
                        help=param_info["description"]
                    )
                    if "explainer_url" in param_info:
                        st.caption(f"Learn more about this parameter: [Watch]({param_info["explainer_url"]})")
                elif param_info["type"] == "boolean":
                    st.checkbox(
                        param,
                        key=f"{id}_{param}_checkbox",
                        value=param_info["default"],
                        help=param_info["description"]
                    )
                elif param_info["type"] == "enum":
                    st.segmented_control(
                        param,
                        key=f"{id}_{param}_segmented_control",
                        options=param_info["options"],
                        default=param_info["default"],
                        selection_mode="single",
                        help=param_info["description"]
                    )
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

                    if default_val != val:
                        if param == "seed":
                            cfg[param] = 42
                        else:
                            cfg[param] = val

                st.session_state.model_configs[id] = cfg
                

    st.divider()
    st.markdown("**Configure Evaluator:**")
    # st.info("Chose structured when, choose free when")
    structured_response_tab, free_response_tab = st.tabs(["Structured Output", "Free Response"])
    possible_labels = None
    with structured_response_tab:
        evaluator_prompt = st.text_area(
            "Describe the criteria for the evaluator:",
            placeholder="I want to test...",
            help="Advice will go here",
            height=150
        )
        if "target" not in st.session_state.experiment_data.columns:
            labels = st.text_input(
                "Enter classification labels for evaluator (comma-separated):", 
                help="""
                The possible classification categories for the evaluator. List the classification labels 
                separated by commas, e.g. "low,medium,high". Spaces will be interpreted as part of the 
                labels,e.g. "too high, too low" will become ["too high", "too low"]"""
            )
            if labels:
                possible_labels = [t.strip().lower() for t in labels.split(",")]
                st.markdown(f"**Evaluator classification labels:** {possible_labels}")
        else:
            st.caption("""
                If the dataset you added in Step 2 includes a target column, 
                the evaluator will mark each response as correct/incorrect. For multiclass
                classification, remove the target column from the dataset."""
            )
            
    start_experiment = st.button("Run Experiment")
    if start_experiment:
        experiment_req_body = {
            "dataset": st.session_state.experiment_data.to_dict(orient="records"),
            "model_configs": st.session_state.model_configs,
            "evaluator_type": "structured",
            "evaluator_config": {
                "evaluator_prompt": evaluator_prompt,
            }
        }
        if possible_labels:
            experiment_req_body["evaluator_config"]["possible_labels"] = possible_labels
        
        file_res = requests.post(
            url=f"{BASE_URL}/experiment/", 
            json=experiment_req_body
        )
        print(file_res.status_code)