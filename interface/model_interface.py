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

    col1, col2 = st.columns(2)
    num_beams = col1.number_input("Number of beams", value=2)
    top_p_sampling = col2.number_input("Nucles (top-p) sampling", value=1)

    with st.spinner("Generating summary..."):
        summary, summary_analysis, score = predict_summary(
            model,
            tokenizer,
            xsum_example["document"],
            num_beams=num_beams,
            analyze_prediction=True,
            top_p_sampling=top_p_sampling
        )
        st.write(f"Summary Beam Score: {score}")
        df_summary = pd.DataFrame(
            summary_analysis,
            columns=[
                "Token (id)" ,
                "Token Prob",
                "Beam idx",
                "Local Entropy",
                "In Source Doc?",
                "Token 1",
                "Token 1 Prob",
                "Token 2",
                "Token 2 Prob",
                "Token 3",
                "Token 3 Prob",
            ]
        )
        st.dataframe(df_summary)
        # st.write(df_summary["Token Prob"].mean())

    st.subheader("Generated Summary")
    st.write(summary)


if __name__ == "__main__":
    render_model_interface()
