import streamlit as st
import pandas as pd
import requests
import altair as alt

BASE_URL = "http://127.0.0.1:8000"

@st.cache_data
def convert_for_download(df):
    """Convert dataframe to csv."""
    return df.to_csv().encode("utf-8")

def fetch_experiment_data():
    """Get experiment data."""
    experiment_results = requests.get(
        url=f"{BASE_URL}/experiments/{st.session_state.experiment_id}/export"
    )
    if experiment_results.ok:   
        data = experiment_results.json()
        df = pd.json_normalize(data["data"])
        return df
    else:
        st.error(f"Loading experiment data failed: {experiment_results.status_code} - {experiment_results.text}")

st.title("Experiment Dashboard")
st.divider()

experiment_id = st.session_state.get("experiment_id")
experiment_status = st.session_state.get("experiment_completed")

# If there is no completed experiment stop execution of this page
if not experiment_id or not experiment_status:
    st.info("Nothing here yet. Start an experiment to populate the dashboard.")
    st.stop()

# Load experiment data and filter out any jobs that failed or were cancelled
exp_df = fetch_experiment_data()
exp_df = exp_df.rename(columns={"id": "job_id"})
valid_rows = exp_df[exp_df["job_status"] == "completed"]
possible_labels = valid_rows["eval_label"].unique()
models = valid_rows["model_id"].unique()

# If more than one model was evaluated, show grouped bar plot
if len(models) > 1:
    with st.container(border=False):
        st.subheader("Results by Model")
        stack = st.toggle("Toggle stacked bars")
        
        # Group counts by model and label
        counts = (
            valid_rows.groupby(["model_id", "eval_label"])
            .size()
            .reset_index(name="count")
        )

        st.bar_chart(
            counts,
            x="eval_label",
            y="count",
            color="model_id",
            x_label="Count",
            y_label="",
            horizontal=True,
            stack=stack,
            height=400,
            use_container_width=True,
        )

# For each model, create a bar chart for its results and a widget to sample responses by category
for id in models:
    col1, col2 = st.columns(2, gap="medium", border=False)
    
    # Get the counts of each label
    counts = valid_rows["eval_label"][valid_rows["model_id"] == id].value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts["percent"] = counts["count"] / counts["count"].sum() * 100

    with col1:
        st.subheader(id)

        # Create a bar chart for this model
        chart = (
            alt.Chart(counts, height=300)
            .mark_bar()
            .encode(
                x=alt.X("count", title="Count"),
                y=alt.Y("label", sort="-x", title=None),
                color=alt.Color("label", legend=None),
                tooltip=[
                    alt.Tooltip("label", title="Class"),
                    alt.Tooltip("count", title="Count"),
                    alt.Tooltip("percent", format=".1f", title="Percent (%)"),
                ],
            )

        )
        st.altair_chart(chart, use_container_width=True)

        # Display metrics underneath chart
        metric_cols = st.columns(len(possible_labels))
        for i, metric_col in enumerate(metric_cols):
            val = counts["percent"][counts["label"] == possible_labels[i]]
            percent = val.iloc[0] if not val.empty else 0
            metric_col.metric(possible_labels[i], f"{percent}%")
    
    # Response sampler   
    with col2:
        label_map = {f"{label} example": label for label in counts["label"].unique()}
        selected_display = st.pills(
            "samples", 
            default=list(label_map.keys())[0], 
            options=list(label_map.keys()), 
            key=f"pills_{id}",
            selection_mode="single",
            label_visibility="collapsed"
        )
        selected_label = label_map[selected_display] if selected_display else None

        # For the selected label, sample a response with that label and display it
        with st.container(height=500):
            if selected_label:
                sample = valid_rows[(valid_rows["model_id"] == id) & (valid_rows["eval_label"] == selected_label)].sample(n=1)

                st.markdown("**Prompt**")
                st.write(f"{sample["user_prompt"].values[0]}")
                st.markdown("**Model Response**")
                st.write(f"{sample["model_response"].values[0]}")
                st.markdown("**Evaluator Justification**")
                st.write(f"{sample["eval_justification"].values[0]}")
   
# View and export full experiment data           
st.divider()
if st.toggle("View experiment data"):
    st.dataframe(exp_df, use_container_width=True)
    
csv = convert_for_download(exp_df)
st.download_button(
    label="Export data",
    data=csv,
    file_name=f"experiment{experiment_id}.csv",
    mime="text/csv",
    icon=":material/download:",
)