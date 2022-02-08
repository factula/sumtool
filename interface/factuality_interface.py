import streamlit as st
from backend.viz_data_loader import load_annotated_data_by_id
from backend.metrics import compute_summary_metrics


@st.experimental_memo
def cache_load_annotated_data_by_id():
    return load_annotated_data_by_id()


annotated_data_by_id = cache_load_annotated_data_by_id()

@st.experimental_memo
def cached_summary_metrics(gt_summary, factuality_data):
    bert_scores = []
    for system in factuality_data:
        bert_score = compute_summary_metrics(
            gt_summary, 
            system["generated_summary"]
        )
        bert_scores.append(
            bert_score
        )
    return bert_scores

def render_factuality_interface():
    st.header("XSUM with Factuality & Faithfulness Labels")

    st.write(f"**Annotated summaries:** {len(annotated_data_by_id)}")
    selected_id = str(
        st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys())
    )

    selected_data = annotated_data_by_id[selected_id]
    st.write("**Ground Truth Summary:**")
    st.write(selected_data["ground_truth_summary"])
    st.write("**Factuality Annotations:**")
    st.table(selected_data["factuality"])
    st.write(cached_summary_metrics(
        selected_data["ground_truth_summary"], 
        selected_data["factuality"]
    ))
    
    st.write("**Faithfulness annotation (TODO: merge overlapping annotations):**")
    st.dataframe(selected_data["faithfulness"])
    st.write("**Document:**")
    st.write(selected_data["document"])


if __name__ == "__main__":
    render_factuality_interface()
