import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(layout="wide")
st.title("LLM Behavior Tester")
st.subheader("Step 1. Select the models you want to test")

res = requests.get(url="http://127.0.0.1:8000/models/")
models = json.loads(res.content)
models_df = pd.json_normalize(models["data"])
models_df = models_df[models_df["id"] != "openrouter/auto"]
cond = (models_df["pricing.prompt"].astype(str).str.strip() == "0") & \
       (models_df["pricing.completion"].astype(str).str.strip() == "0")
models_df["free"] = cond

only_free = st.toggle("Only show free models", value=False)
display_df = models_df[models_df["free"]] if only_free else models_df
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

st.subheader("Step 2. Create dataset")
st.write("Generate prompts")
text_input = st.text_input(
    "Enter some text",
    placeholder="Describe your idea",
)

editor = st.data_editor(display_df)

# filter_container = st.container()

# with filter_container:
#     filter_columns = st.multiselect("Filter models by:", display_df.columns)
#     for col in filter_columns:
#         left, right = st.columns((1, 50))
#         left.write("↳")
#         user_text_input = right.text_input(
#             f"Search {col}",
#         )
#         if user_text_input:
#             display_df = display_df[display_df[col].str.contains(user_text_input)]

#     model_picker = st.dataframe(
#         display_df,
#         # column_config=column_configuration,
#         use_container_width=True,
#         hide_index=True,
#         on_select="rerun",
#         selection_mode="multi-row",
#     )

#     selected_models = model_picker.selection.rows
#     filtered_df = models_df.iloc[selected_models]

#     st.write("Selected models:")
#     st.dataframe(filtered_df["id"], hide_index=True)