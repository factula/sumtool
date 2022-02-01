import streamlit as st
from datasets import load_dataset
from collections import defaultdict
import pandas as pd

def collapse_factuality_annotations(annotations):
    collapsed = []
    for system in ["BERTS2S", "TranS2S", "TConvS2S", "PtGen"]:
        summary = None
        sum_factual = 0
        total = 0
        for annotation in annotations:
            if annotation["system"] == system:
                summary = annotation["summary"]
                total += 1
                sum_factual += annotation["is_factual"]
        
        if summary is not None:
            collapsed.append({
                "system": system,
                "generated_summary": summary,
                "mean_worker_factuality_score": sum_factual / total
            })

    return collapsed

@st.experimental_memo
def load_annotated_data_by_id():
    # Load datasets
    # TODO: use solution from https://github.com/cs6741/summary-analysis/issues/2 to align data
    data_xsum = load_dataset("xsum")
    data_factuality = load_dataset("xsum_factuality")["train"]
    data_faithfulness = load_dataset("xsum_factuality", "xsum_faithfulness")["train"]
    
    annotated_data_by_id = {}
    for annotation in data_factuality:
        bbcid = str(annotation["bbcid"])
        if bbcid in annotated_data_by_id:
            annotated_data_by_id[bbcid]["factuality"].append(annotation)
        else:
            annotated_data_by_id[bbcid] = {
                "id": bbcid,
                "factuality": [annotation],
                "faithfulness": []
            }

    for annotation in data_faithfulness:
        bbcid = str(annotation["bbcid"])
        if bbcid in annotated_data_by_id:
            annotated_data_by_id[bbcid]["faithfulness"].append(annotation)
        else:
            annotated_data_by_id[bbcid] = {
                "id": bbcid,
                "faithfulness": [annotation],
                "factuality": []
            }
    
    for key in annotated_data_by_id.keys():
        annotated_data_by_id[key]["factuality"] = collapse_factuality_annotations(annotated_data_by_id[key]["factuality"])


    for row in data_xsum["test"]:
        if row["id"] in annotated_data_by_id:
            annotated_data_by_id[row["id"]]["summary"] = row["summary"]
            annotated_data_by_id[row["id"]]["document"] = row["document"]

    return annotated_data_by_id

annotated_data_by_id = load_annotated_data_by_id()

# UI
st.header("CS6741 - Summary Analysis")
st.subheader("XSUM with Factuality & Faithfulness Labels")

st.write(f"**Annotated summaries:** {len(annotated_data_by_id)}")
selected_id = str(st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys()))

selected_data = annotated_data_by_id[selected_id]
st.write("**Ground Truth Summary:**")
st.write(selected_data['summary'])
st.write("**Factuality Annotations:**")
st.table(selected_data["factuality"])

# st.write("**Faithfulness annotation (TODO: merge overlapping annotations):**")
# st.dataframe(selected_data['faithfulness'])
st.write("**Document:**")
st.write(selected_data['document'])
