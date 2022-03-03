import streamlit as st
from sumtool import generate_xsum_summary, storage
from backend.viz_data_loader import load_annotated_data_by_id
import pandas as pd

annotated_data_by_id = load_annotated_data_by_id()


@st.experimental_singleton
def cache_load_summarization_model_and_tokenizer():
    return generate_xsum_summary.load_summarization_model_and_tokenizer()


def render_model_interface():

    st.header("Generate a Summary")

    # Select/Input source document
    selected_id = str(
        st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys())
    )
    selected_data = annotated_data_by_id[selected_id]
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

    cache_summaries = storage.get_summaries("xsum", "facebook-bart-large-xsum")
    cache_summary_values = [*cache_summaries.values()][0]
    if selected_id in cache_summary_values:
        predicted_summary = cache_summary_values[selected_id]["summary"][0]
    else:
        with st.spinner("Loading pre-trained model"):
            model, tokenizer = cache_load_summarization_model_and_tokenizer()

        with st.spinner("Generating summary..."):
            predicted_summary = generate_xsum_summary.generate_summaries(
                model, tokenizer, source
            )[0]
        storage.store_model_summaries(
            "xsum",
            model.config.name_or_path,
            model.config.to_dict(),
            {selected_id: predicted_summary},
        )

    # Output summarization
    st.subheader("Predicted Summary with BART XSUM Model")
    st.write(predicted_summary)


if __name__ == "__main__":
    render_model_interface()
