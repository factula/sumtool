import streamlit as st
import annotated_text
import pandas as pd
from backend.viz_data_loader import load_annotated_data_by_id
from utils.faithfulness_annotations import (
    annotation_overlap,
    annotation_merge,
    annotation_render,
    annotation_text,
    annotation_color
)


@st.experimental_memo
def cache_load_annotated_data_by_id():
    return load_annotated_data_by_id()


annotated_data_by_id = cache_load_annotated_data_by_id()

def render_faithfulness_interface():
    st.header("XSUM with Faithfulness Annotations")

    st.write(f"**# of Annotated summaries:** {len(annotated_data_by_id)}")
    selected_id = str(
        st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys())
    )

    render_ann_presence = st.checkbox("Distinguish annotation presence", value=False)
    render_ann_halltype = st.checkbox("Distinguish annotation type", value=False)

    annotated_text.annotated_text(
        *annotation_render(
            [
                ["Hallucination type 0 (intrinsic?).", {(0, "1"), (0, "2"), (0, "3")}],
                ["When only some annotators notice it.", {(0, "1"), (0, "2")}],
            ],
            annotation_text,
            annotation_color,
            max_count=3,
            render_presence=render_ann_presence,
            render_type=render_ann_halltype,
        )
    )
    annotated_text.annotated_text(
        *annotation_render(
            [
                ["Hallucination type 1 (extrinsic?).", {(1, "1"), (1, "2"), (1, "3")}],
                ["When only some annotators notice it.", {(1, "1"), (1, "2")}],
            ],
            annotation_text,
            annotation_color,
            max_count=3,
            render_presence=render_ann_presence,
            render_type=render_ann_halltype,
        )
    )
    annotated_text.annotated_text(
        *annotation_render(
            [
                [
                    "Hallucination type mostly 0.",
                    {(0, "1"), (0, "2"), (0, "3"), (0, "4"), (1, "5"), (1, "6")},
                ],
                [
                    "When only some annotators notice it.",
                    {(0, "1"), (0, "2"), (1, "5")},
                ],
            ],
            annotation_text,
            annotation_color,
            max_count=6,
            render_presence=render_ann_presence,
            render_type=render_ann_halltype,
        )
    )
    annotated_text.annotated_text(
        *annotation_render(
            [
                [
                    "Hallucination type mostly 1.",
                    {(1, "1"), (1, "2"), (1, "3"), (1, "4"), (0, "5"), (0, "6")},
                ],
                [
                    "When only some annotators notice it.",
                    {(1, "1"), (1, "2"), (0, "5")},
                ],
            ],
            annotation_text,
            annotation_color,
            max_count=6,
            render_presence=render_ann_presence,
            render_type=render_ann_halltype,
        )
    )

    st.write("---")

    selected_data = annotated_data_by_id[selected_id]
    selected_annotations = pd.DataFrame(selected_data["faithfulness"])

    # summarize annotations for each model summary
    # TODO present these in an ordered way
    # TODO add checkboxes for which workers to include???
    for (g_model, g_summary), g_annotations in selected_annotations.groupby(
        ["system", "summary"]
    ):
        # fields in annotations df:
        # bbcid, system, summary, hallucination_type,
        # hallucinated_span_start, hallucinated_span_end, worker_id
        st.write(f"**{g_model}:**")
        # if g_model!='Gold':
        #     g_factuality = selected_factuality[
        #         selected_factuality['system']==g_model
        #     ].iloc[0]['mean_worker_factuality_score']
        #     st.write(f'mean_worker_factuality_score={str(g_factuality)}')
        # st.write(g_annotations)
        ann = [[g_summary, set()]]
        for _, r in g_annotations[g_annotations["hallucination_type"] != -1].iterrows():
            h_begin = r["hallucinated_span_start"] - 1
            h_end = r["hallucinated_span_end"] - 1
            ann = annotation_overlap(
                ann,
                h_begin,
                h_end,
                (r["hallucination_type"], r["worker_id"]),
                # (r['hallucination_type'],),
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
        st.write("")

    st.write("**Document:**")
    st.write(selected_data["document"])

    # st.write("**Ground Truth Summary:**")
    # st.write(selected_data['ground_truth_summary'])

    st.write("**Faithfulness Annotation source data:**")
    st.write(selected_annotations)


if __name__ == "__main__":
    render_faithfulness_interface()
