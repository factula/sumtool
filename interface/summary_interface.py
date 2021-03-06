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

annotated_data_by_id = load_annotated_data_by_id()


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
    selected_faithfulness = pd.DataFrame(selected_data["faithfulness_data"])
    df_factuality = pd.DataFrame(selected_data["factuality_data"].values())

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

    # Settings
    selected_summsrc = str(
        st.selectbox(
            "Select summary source", options=["Gold"] + sorted(df_factuality.index)
        )
    )
    min_span_len = st.number_input("Minimum highlight length", min_value=1, value=3)
    filter_overlaps = st.checkbox("Filter out overlaps", value=True)

    # Save the summary and source texts
    selected_summsrc_summ = selected_faithfulness[
        selected_faithfulness.system == selected_summsrc
    ].iloc[0]["summary"]
    selected_summsrc_src = selected_data["document"]

    # Do a search for spans in summary that are contained within the document
    linked_spans = {}
    # TODO can definitely be more efficient, this is extremely naive

    def searching_spans_gen(selected_summsrc_summ, min_span_len):
        searching_spans = [(0, len(selected_summsrc_summ))]
        # TODO use actual proper substring/token recognition
        last_end = len(selected_summsrc_summ)
        while last_end > 0:
            new_spans = [
                (i.start() + 1, i.end() - 1)
                for i in re.finditer(
                    r"[ \".,'\\/][^ \".,'\\/].*[ \".,'\\/]",
                    selected_summsrc_summ[:last_end],
                    overlapped=True,
                )
            ] + [
                (i.start(), i.end() - 1)
                for i in re.finditer(
                    r"^[^ \".,'\\/].*[ \".,'\\/]",
                    selected_summsrc_summ[:last_end],
                    overlapped=True,
                )
            ]
            new_spans = sorted(set(new_spans), key=lambda x: x[0])
            searching_spans += [e for e in new_spans if e[1] - e[0] >= min_span_len]
            if len(new_spans) == 0:
                last_end = 0
                break
            last_end -= 1
        searching_spans = sorted(set(searching_spans), key=lambda x: x[0])
        return searching_spans

    searching_spans = searching_spans_gen(selected_summsrc_summ, min_span_len)
    # scan through every substring/token span for matches
    for (i, j) in searching_spans:

        def token_search(token, text):
            match_list = [
                (i.start() + 1, i.end() - 1)
                for i in re.finditer(
                    r"[ \".,'\\/]" + token + r"[ \".,'\\/]",
                    text,
                    re.IGNORECASE,
                )
            ] + [
                (i.start(), i.end() - 1)
                for i in re.finditer(
                    r"^" + token + r"[ \".,'\\/]",
                    text,
                    re.IGNORECASE,
                )
            ]
            match_list = sorted(set(match_list), key=lambda x: x[0])
            return match_list

        match_list_src = token_search(selected_summsrc_summ[i:j], selected_summsrc_src)
        match_list_summ = token_search(
            selected_summsrc_summ[i:j], selected_summsrc_summ
        )
        if len(match_list_src) > 0 and len(match_list_summ) > 0:
            linked_spans[(i, j)] = {"src": match_list_src, "summ": match_list_summ}
    # pick what span matches to prioritize highlights, based on descending span length
    searching_matches = sorted(linked_spans, key=lambda x: x[1] - x[0], reverse=True)
    # filter out overlapping highlights
    # TODO this is slightly buggy considering there can be duplicate tokens in summ
    highlights = []
    for s in searching_matches:
        if (not filter_overlaps) or all(
            [(s[1] <= h[0] or s[0] >= h[1]) for h in highlights]
        ):
            highlights += [s]
    # render the search findings
    cm_color = iter(cm.rainbow(np.linspace(0, 1, len(highlights))))
    ann_links_summ = [[selected_summsrc_summ, set()]]
    ann_links_src = [[selected_summsrc_src, set()]]
    for i in range(len(highlights)):
        tag = f"S{i}"
        c = next(cm_color)
        for j in range(len(linked_spans[highlights[i]]["summ"])):
            ann_links_summ = annotation_overlap(
                ann_links_summ,
                linked_spans[highlights[i]]["summ"][j][0],
                linked_spans[highlights[i]]["summ"][j][1],
                (tag, tuple(c)),
            )
        for j in range(len(linked_spans[highlights[i]]["src"])):
            ann_links_src = annotation_overlap(
                ann_links_src,
                linked_spans[highlights[i]]["src"][j][0],
                linked_spans[highlights[i]]["src"][j][1],
                (tag, tuple(c)),
            )

    def link_tag(feats):
        return list(feats)[0][0]

    def link_color(feats, rp, rt, max_overlap):
        f = list(feats)[0][1]
        m = 156
        return f"rgba({m*f[0]}, {m*f[1]}, {m*f[2]}, {f[3]})"

    annotated_text.annotated_text(
        *annotation_render(
            ann_links_summ,
            link_tag,
            link_color,
            max_overlap=1,
        )
    )
    st.write("")
    annotated_text.annotated_text(
        *annotation_render(
            ann_links_src,
            link_tag,
            link_color,
            max_overlap=1,
        )
    )


if __name__ == "__main__":
    render_summary_interface()
