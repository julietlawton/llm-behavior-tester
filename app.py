import streamlit as st
import pandas as pd
import requests
import json
import time

st.set_page_config(layout="wide")

# key_res = requests.get(url="http://127.0.0.1:8000/check_key/")
# st.write(json.dumps(key_res.json(), indent=2))

st.title("LLM Behavior Tester")
st.subheader("Step 1. Select the models you want to test")

res = requests.get(url="http://127.0.0.1:8000/models/")
models = json.loads(res.content)
models_df = pd.json_normalize(models["data"])
models_df = models_df[models_df["id"] != "openrouter/auto"]

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

st.write("Selected models:")
st.dataframe(filtered_df["id"], hide_index=True)
experiment_df = None

if selected_models:
    st.subheader("Step 2. Create dataset")
    upload_dataset_tab, generate_dataset_tab = st.tabs(["Upload Dataset", "Generate Dataset"])
    
    with upload_dataset_tab:
        uploaded_dataset = st.file_uploader(
            "Choose a CSV or JSON file:",
            type=["csv", "json"],
            accept_multiple_files=False
        )
        if uploaded_dataset is not None:
            if uploaded_dataset.type == "csv":
                user_df = pd.read_csv(uploaded_dataset)
                st.data_editor(user_df, num_rows="dynamic")
            else:
                user_df = pd.read_json(uploaded_dataset)
                st.data_editor(user_df, num_rows="dynamic")
            
            experiment_df = user_df

    with generate_dataset_tab:
        default_model = "openai/gpt-oss-20b:free"
        all_models = models_df["id"][models_df["id"] != default_model].values
        model_options = [default_model] + list(all_models)
        num_prompts = st.number_input("Number of prompts to generate:", value=10, min_value=1, max_value=1000)

        model_selection = st.selectbox(
            "Prompt generation model:",
            model_options,
        )
        system_prompt_toggle = st.toggle("Include system prompt", value=False)
        if system_prompt_toggle:
            system_prompt = st.text_area(
                "Add a system prompt. This prompt will be used for ",
                placeholder="Add system prompt",
                help="Advice",
                height = 150
            )
        dataset_gen_prompt = st.text_area(
            "Describe the experiment you want to run:",
            placeholder="Describe your idea",
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
                prompt_res = requests.post(
                    url="http://127.0.0.1:8000/generate_prompts/", 
                    params={
                        "model_id": model_selection,
                        "experiment_description": dataset_gen_prompt, 
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
                    experiment_df = generated_df

if experiment_df is not None:
    st.subheader("Step 3. Set Up Experiment")