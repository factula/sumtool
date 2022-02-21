import streamlit as st
import torch
from sumtool import predict_xsum_summary
from backend.viz_data_loader import load_annotated_data_by_id
import pandas as pd

annotated_data_by_id = load_annotated_data_by_id()


def render_model_interface():

    st.header("Generate a Summary")

    # Load the pretrained model
    model, tokenizer = predict_xsum_summary.load_summarization_model_and_tokenizer()

    # Select/Input source document
    selected_id = str(
        st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys())
    )
    selected_data = annotated_data_by_id[selected_id]
    source = selected_data["document"]
    st.subheader("Source Document")
    st.write(source)

    # Summarize source document
    predicted_summary = predict_xsum_summary.predict_summary(model, tokenizer, source)

    # Ground Truth Summary
    selected_faithfulness = pd.DataFrame(selected_data["faithfulness_data"])
    g_summary = (
        selected_faithfulness[selected_faithfulness.system == "Gold"].iloc[0].summary
    )

    st.subheader("Ground Truth Summary")
    st.write(g_summary)

    # Output summarization
    st.subheader("Predicted Summary with BART XSUM Model")
    st.write(predicted_summary)


if __name__ == "__main__":
    render_model_interface()
