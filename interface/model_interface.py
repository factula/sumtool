import streamlit as st
from sumtool import storage
from backend.viz_data_loader import load_annotated_data_by_id
import pandas as pd

cache_summaries = storage.get_summaries("xsum", "facebook-bart-large-xsum")
cache_keys = cache_summaries.keys()
annotated_data_by_id = load_annotated_data_by_id()
filtered_annotated_data_by_id = {
    k: annotated_data_by_id[k]
    for k in cache_summaries.keys()
    if k in annotated_data_by_id
}


def render_model_interface():

    st.header("Generate a Summary with BART XSUM Model")

    # Select/Input source document
    selected_id = str(
        st.selectbox(
            "Select entry by bbcid", options=filtered_annotated_data_by_id.keys()
        )
    )
    selected_data = filtered_annotated_data_by_id[selected_id]
    source = selected_data["document"]
    st.subheader("Source Document")
    st.write(source)

    # Ground Truth Summary
    selected_faithfulness = pd.DataFrame(selected_data["faithfulness_data"])
    g_summary = (
        selected_faithfulness[selected_faithfulness.system == "Gold"].iloc[0].summary
    )
    st.subheader("Ground Truth Summary")
    st.write(g_summary)

    # Output summarization
    predicted_summary = cache_summaries[selected_id]["summary"][0]
    st.subheader("Predicted Summary")
    st.write(predicted_summary)


if __name__ == "__main__":
    render_model_interface()
