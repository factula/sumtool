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
import numpy as np
import regex as re
# Colormap for varying-number annotations
from matplotlib.pyplot import cm


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
            max_overlap=3,
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

    st.write("---")

    st.subheader("Summary-Source Links")

    selected_summsrc = str(
        st.selectbox(
            "Select summary source", options=["Gold"] + sorted(df_factuality.index)
        )
    )
    # Save the summary and source texts
    selected_summsrc_summ = selected_faithfulness[
        selected_faithfulness.system == selected_summsrc
    ].iloc[0]["summary"]
    selected_summsrc_src = selected_data["document"]

    # Do a search for spans (length>=2) in summary that are contained within the document
    linked_spans = {}
    # TODO can definitely be more efficient, this is extremely lazy
    searching_spans = [(0, len(selected_summsrc_summ))]
    # TODO use actual proper substring/token recognition
    last_end = len(selected_summsrc_summ)
    while last_end > 0:
        new_spans = [
            (i.start() + 1, i.end() - 1)
            for i in re.finditer(
                r"[ \".,'].*[ \".,']", selected_summsrc_summ[:last_end], overlapped=True
            )
        ]
        searching_spans += new_spans
        searching_spans = [e for e in searching_spans if e[1] - e[0] >= 3]
        if len(new_spans) == 0:
            break
        last_end = new_spans[-1][0] + 1
    searching_spans = sorted(set(searching_spans), key=lambda x: x[0])
    # scan through every substring/token span for matches
    for (i, j) in searching_spans:
        match_list = re.finditer(
            selected_summsrc_summ[i:j], selected_summsrc_src, re.IGNORECASE
        )
        match_list = [(m.start(0), m.end(0)) for m in match_list]
        if len(match_list) > 0:
            linked_spans[(i, j)] = match_list
    # pick what span matches to prioritize highlighting
    # TODO: currently prioritizing span length solely, excluding shorter spans that overlap
    searching_matches = sorted(linked_spans, key=lambda x: x[1] - x[0], reverse=True)
    # filter out overlapping highlights
    highlights = []
    for s in searching_matches:
        if all([(s[1] <= h[0] or s[0] >= h[1]) for h in highlights]):
            highlights += [s]
    # highlights = [(summ_begin, summ_end)] where each tuple is a key in linked_spans
    # linked_spans[(summ_begin, summ_end)] = [(src_begin1, src_end1), (src_begin2, src_end2)]
    # TODO: render the search findings
    st.write(
        [
            (
                selected_summsrc_summ[k[0] : k[1]],
                len(linked_spans[k]),
                str(k),
                str(linked_spans[k]),
            )
            for k in highlights
        ]
    )
    cm_color = iter(cm.rainbow(np.linspace(0, 1, len(highlights))))
    for i in range(len(highlights)):
        c = next(cm_color)
        st.write(str(c))
    st.write(selected_summsrc_summ)


if __name__ == "__main__":
    render_summary_interface()
