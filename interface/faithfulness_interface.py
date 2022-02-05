import streamlit as st
import pandas as pd
from backend.viz_data_loader import load_annotated_data_by_id
import annotated_text

@st.experimental_memo
def cache_load_annotated_data_by_id():
    return load_annotated_data_by_id()

annotated_data_by_id = cache_load_annotated_data_by_id()

def render_faithfulness_interface():
    st.header("XSUM with Faithfulness Annotations")

    st.write(f"**# of Annotated summaries:** {len(annotated_data_by_id)}")
    selected_id = str(st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys()))

    st.write("Hallucinations in document summarization.")
    st.write("Hallucination type 0 looks like this.")
    st.write("Hallucination type 1 looks like this.")

    selected_data = annotated_data_by_id[selected_id]
    selected_annotations = pd.DataFrame(selected_data['faithfulness'])

    # st.write("**Document:**")
    # st.write(selected_data['document'])

    # st.write("**Ground Truth Summary:**")
    # st.write(selected_data['ground_truth_summary'])

    # summarize annotations for each model summary
    ann_colors = {
        -1: '#d55',
        0: '#5d5',
        1: '#55d'
    }
    for (g_model, g_summary), g_annotations in selected_annotations.groupby(['system', 'summary']):
        # fields in annotations df:
        # bbcid, system, summary, hallucination_type,
        # hallucinated_span_start, hallucinated_span_end, worker_id
        st.write(f'***{g_model}:***')
        st.write(g_annotations[['hallucination_type', 'hallucinated_span_start', 'hallucinated_span_end', 'worker_id']])
        for _, r in g_annotations[
            g_annotations['hallucinated_span_start']!=g_annotations['hallucinated_span_end']
        ].iterrows():
            h_begin = r['hallucinated_span_start']
            h_end = r['hallucinated_span_end']
            h_type = r['hallucination_type']
            st.write()
            annotated_text.annotated_text(
                g_summary[:h_begin],
                (g_summary[h_begin:h_end], f'{h_type}', ann_colors[h_type]),
                g_summary[h_end:],
            )

    st.write("**Factuality Annotations:**")
    st.table(selected_data["factuality"])

if __name__ == "__main__":
    render_hallucination_interface()
