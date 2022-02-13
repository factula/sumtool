import streamlit as st
from backend.viz_data_loader import load_annotated_data_by_id
from utils.faithfulness_annotations import (
    annotation_overlap,
    annotation_color,
    annotation_text,
    annotation_merge,
    annotation_render,
    render_faithfulness_annotation_legend,
)
import annotated_text
import pandas as pd


@st.experimental_memo
def cache_load_annotated_data_by_id():
    return load_annotated_data_by_id()


annotated_data_by_id = cache_load_annotated_data_by_id()


def render_summary_with_annotations(
    faithfulness_annotations,
    render_ann_presence,
    render_ann_halltype,
    factuality_annotations=None,
):
    g_summary = faithfulness_annotations.iloc[0].summary
    ann = [[g_summary, set()]]
    for _, r in faithfulness_annotations[
        faithfulness_annotations["hallucination_type"] != -1
    ].iterrows():
        h_begin = r["hallucinated_span_start"] - 1
        h_end = r["hallucinated_span_end"] - 1
        ann = annotation_overlap(
            ann,
            h_begin,
            h_end,
            (r["hallucination_type"], r["worker_id"]),
        )
    ann = annotation_merge(ann)
    annotated_text.annotated_text(
        *annotation_render(
            ann,
            annotation_text,
            annotation_color,
            max_count=3,
            render_presence=render_ann_presence,
            render_type=render_ann_halltype,
        )
    )
    st.write("\n")


def render_summary_interface():
    st.header("Explore Summarization Datasets")
    # hard-coded XSUM for now
    st.write(
        f"""
**Selected dataset:** XSUM with Factuality & Faithfulness Annotations (TODO: reference paper) \\
**Annotated summaries:** {len(annotated_data_by_id)}

**TODO:** _aggregated dataset statistics_
    """
    )
    selected_id = str(
        st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys())
    )

    st.write("**Annotations are colored according to the type of hallucination.**")
    col1, col2 = st.columns(2)
    render_ann_presence = col1.checkbox("Visualize annotation strength", value=False)
    render_ann_halltype = col2.checkbox(
        "Visualize annotation type disagreement", value=False
    )

    render_faithfulness_annotation_legend(render_ann_presence, render_ann_halltype)

    st.write("---")

    selected_data = annotated_data_by_id[selected_id]
    selected_faithfulness = pd.DataFrame(selected_data["faithfulness"])
    df_factuality = pd.DataFrame(selected_data["factuality"])

    if "system" in df_factuality.columns:
        df_factuality.set_index("system", inplace=True)

    st.subheader("Ground Truth Summary")
    render_summary_with_annotations(
        selected_faithfulness[selected_faithfulness.system == "Gold"],
        render_ann_presence,
        render_ann_halltype,
    )

    st.subheader("Generated Summaries")

    for (g_model, g_summary), g_annotations in (
        selected_faithfulness[selected_faithfulness.system != "Gold"]
        .sort_values("system")
        .groupby(["system", "summary"])
    ):
        col1, col2 = st.columns([25, 75])
        factuality_score = (
            round(df_factuality.loc[g_model].mean_worker_factuality_score, 2)
            if g_model in df_factuality.index
            else "Missing"
        )
        col1.write(
            f"""
**{g_model}** \\
_factuality score:_ {factuality_score} \\
_rogue1 score:_ **TODO** \\
_rogue2 score:_ **TODO** \\
_bert score:_ **TODO**
        """
        )
        with col2:
            render_summary_with_annotations(
                g_annotations, render_ann_presence, render_ann_halltype
            )

    st.subheader("Source Document")
    st.write(selected_data["document"])


if __name__ == "__main__":
    render_summary_interface()
