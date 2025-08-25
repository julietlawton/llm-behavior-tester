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

selected_models = model_picker.selection.rows
filtered_df = models_df.iloc[selected_models]

if selected_models:
    st.markdown("**Selected models:**")
    st.dataframe(filtered_df["id"], hide_index=True)
experiment_df = None

# if selected_models:
if True:
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
                user_df = pd.read_csv(uploaded_dataset)
                st.data_editor(user_df, num_rows="dynamic")
            else:
                user_df = pd.read_json(uploaded_dataset)
                st.data_editor(user_df, num_rows="dynamic")
            st.write(f"**Total items:** {len(user_df)}")
            experiment_df = user_df

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
        system_prompt_toggle = st.toggle("Include system prompt", value=False)
        if system_prompt_toggle:
            system_prompt = st.text_area(
                "Add a system prompt:",
                placeholder="You are a helpful assistant...",
                help="This prompt will be included with every user prompt.",
                height = 150
            )
        dataset_gen_prompt = st.text_area(
            "Describe the experiment you want to run:",
            placeholder="I want to test...",
            help="Advice will go here",
            height=150
        )

        generate_clicked = st.button(
            "Generate Prompts",
            icon=":material/wand_stars:",
        )

        if generate_clicked:
            if not dataset_gen_prompt.strip():
                warning_placeholder = st.empty()
                with warning_placeholder:
                    st.warning("Please enter an experiment description first.")
                time.sleep(3)
                warning_placeholder.empty()
            else:
                with st.spinner("Generating prompts...", show_time=True):
                    prompt_res = requests.post(
                        url=f"{BASE_URL}/generate/prompts/", 
                        json={
                            "model_id": model_selection,
                            "experiment_description": dataset_gen_prompt, 
                        },
                        params={
                            "n": num_prompts
                        }
                    )
                try:
                    prompt_res.raise_for_status()
                    data = prompt_res.json()
                except requests.HTTPError as e:
                    st.error(f"Request failed: {e}\n{prompt_res.text}")
                else:
                    generated_df = pd.DataFrame(data["items"])
                    st.data_editor(generated_df, num_rows="dynamic")
                    if "errors" in data:
                        n_errors = len(data["errors"])
                        st.badge(
                            f"{n_errors} job{"" if n_errors == 1 else "s"} failed", 
                            icon=":material/warning:", 
                            color="orange"
                        )
                        st.warning(data["errors"][0]["message"])
                    else:
                        st.badge("All jobs finished successfully", icon=":material/check:", color="green")
                    st.write(f"**Total items generated:** {data["total"]} **Job cost:** ${data["usage"]}")
                    experiment_df = generated_df

# if experiment_df is not None:
if True:
    st.subheader("Step 3. Set Up Experiment")
    st.markdown("**Configure Models:** (Optional)")

    for id in filtered_df["id"].values:
        with st.expander(id, expanded=False):
            param_options = [
                p for p in filtered_df["supported_parameters"][filtered_df["id"] == id].values[0] 
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
            st.button("Save Settings", key=f"{id}_save_button")
                

    st.divider()
    st.markdown("**Configure Evaluator:**")
    # st.info("Chose structured when, choose free when")
    # structured_response_tab, free_response_tab = st.tabs(["Structured Output", "Free Response"])
            
    # start_experiment = st.button("Run Experiment")
    # if start_experiment and experiment_df is not None:

    #     file_res = requests.post(
    #         url=f"{BASE_URL}/experiment/", 
    #         json={
    #             "dataset": experiment_df.to_dict(orient="records")
    #         }
    #     )
    #     st.write(file_res.status_code, file_res.text)
