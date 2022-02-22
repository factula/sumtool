import streamlit as st
from sumtool.predict_xsum_summary import predict_summary, load_summarization_model_and_tokenizer
from backend.viz_data_loader import load_annotated_data_by_id
import pandas as pd

annotated_data_by_id = load_annotated_data_by_id()


def render_model_interface():
    st.header("Generate a Summary")

    model, tokenizer = load_summarization_model_and_tokenizer()

    selected_id = str(
        st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys())
    )
    xsum_example = annotated_data_by_id[selected_id]

    st.subheader("Document")
    st.write(xsum_example["document"])

    num_beams = st.number_input("Number of beams", value=2)
    
    with st.spinner("Generating summary..."):
        summary, summary_analysis, score = predict_summary(
            model,
            tokenizer,
            xsum_example["document"],
            num_beams=num_beams,
            analyze_prediction=True
        )
        st.write(f"Summary score: {score}")
        st.dataframe(pd.DataFrame(
            summary_analysis,
            columns=[
                "Selected Token (id)" ,
                "Prob",
                "Beam idx",
                "Local Entropy",
                "In Source Doc?",
                "B1 Top 1",
                "B1 Top 1 Prob",
                "B1 Top 2",
                "B1 Top 2 Prob",
                "B1 Top 3",
                "B1 Top 3 Prob",
            ]
        ))

    st.subheader("Generated Summary")
    st.write(summary)


if __name__ == "__main__":
    render_model_interface()
