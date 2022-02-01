import streamlit as st
from backend.viz_data_loader import load_annotated_data_by_id

@st.experimental_memo
def cache_load_annotated_data_by_id():
    return load_annotated_data_by_id()

annotated_data_by_id = cache_load_annotated_data_by_id()

# UI
st.header("CS6741 - Summary Analysis")
st.subheader("XSUM with Factuality & Faithfulness Labels")

st.write(f"**Annotated summaries:** {len(annotated_data_by_id)}")
selected_id = str(st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys()))

selected_data = annotated_data_by_id[selected_id]
st.write("**Ground Truth Summary:**")
st.write(selected_data['ground_truth_summary'])
st.write("**Factuality Annotations:**")
st.table(selected_data["factuality"])

st.write("**Faithfulness annotation (TODO: merge overlapping annotations):**")
st.dataframe(selected_data['faithfulness'])
st.write("**Document:**")
st.write(selected_data['document'])
