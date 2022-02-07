import streamlit as st
import pandas as pd
from backend.viz_data_loader import load_annotated_data_by_id
import annotated_text

@st.experimental_memo
def cache_load_annotated_data_by_id():
    return load_annotated_data_by_id()

annotated_data_by_id = cache_load_annotated_data_by_id()

def annotation_overlap(ann, n_start, n_end, n_details):
    if n_start>n_end:
        raise ValueError()
    # find indexes of beginning and end chunk presence
    prevstart_i = 0
    annstart_i = 0
    while annstart_i<len(ann) and prevstart_i+len(ann[annstart_i][0])<=n_start:
        prevstart_i += len(ann[annstart_i][0])
        annstart_i += 1
    prevend_i = prevstart_i
    annend_i = annstart_i
    while annend_i<len(ann) and prevend_i+len(ann[annend_i][0])<=n_end:
        prevend_i += len(ann[annend_i][0])
        annend_i += 1
    # check if we only need to modify one chunk
    if annstart_i>=len(ann):
        pass
    elif annstart_i==annend_i:
        if n_end>0:
            ann.insert(annstart_i, [ann[annstart_i][0], ann[annstart_i][1].copy()])
            ann.insert(annstart_i, [ann[annstart_i][0], ann[annstart_i][1].copy()])
            ann[annstart_i][0] = ann[annstart_i][0][:n_start-prevstart_i]
            ann[annstart_i+1][0] = ann[annstart_i+1][0][n_start-prevstart_i:n_end-prevend_i]
            ann[annstart_i+2][0] = ann[annstart_i+2][0][n_end-prevend_i:]
            ann[annstart_i+1][1].add(n_details)
    else:
        # this needs to be done across multiple chunks
        # add to beginning chunk
        if n_start<=prevstart_i:
            ann[annstart_i][1].add(n_details)
        else:
            # split out a new beginning subchunk
            ann.insert(annstart_i, [ann[annstart_i][0], ann[annstart_i][1].copy()])
            ann[annstart_i][0] = ann[annstart_i][0][:n_start-prevstart_i]
            ann[annstart_i+1][0] = ann[annstart_i+1][0][n_start-prevstart_i:]
            ann[annstart_i+1][1].add(n_details)
            annstart_i += 1
            annend_i += 1
        # add to intermediate chunk(s)
        for i in range(annstart_i+1, annend_i-1):
            ann[i][1].add(n_details)
        # add to end chunk
        if annend_i==len(ann) or n_end==prevend_i+len(ann[annend_i-1][0]):
            if annend_i-1!=annstart_i:
                ann[annend_i-1][1].add(n_details)
        else:
            # split out a new end subchunk
            ann.insert(annend_i, [ann[annend_i][0], ann[annend_i][1].copy()])
            ann[annend_i][0] = ann[annend_i][0][:n_end-prevend_i]
            ann[annend_i+1][0] = ann[annend_i+1][0][n_end-prevend_i:]
            if annend_i-1!=annstart_i:
                ann[annend_i-1][1].add(n_details)
            ann[annend_i][1].add(n_details)
    # clean out the chunks
    ann = [e for e in ann if len(e[0])!=0]
    return ann

# Combine sequential chunks of annotation list that have identical tagging
def annotation_merge(ann):
    if len(ann)==0:
        return ann
    # keep appending as long as next piece is identical to last piece
    new_ann = [ann[0]]
    for a in ann[1:]:
        if a[1]==new_ann[-1][1]:
            new_ann[-1][0]+=a[0]
        else:
            new_ann.append(a)
    return new_ann

def annotation_render(ann, func_text, func_color):
    return [
        (e[0]) if (len(e[1])==0) else (e[0], func_text(e[1]), func_color(e[1]))
        for e in ann
    ]

def annotation_text(feats):
    # Assume from the XSum data that there are up to 3 workers, with tags of 0/1
    tag_avg = [int(i[0]) for i in feats]
    tag_avg = sum(tag_avg)/len(tag_avg)
    return str(int(tag_avg))

def annotation_color(feats):
    # Assume from the XSum data that there are up to 3 workers, with tags of 0/1
    ann_colors = {
        0: '#5d5',
        1: '#55d'
    }
    tag_avg = [int(i[0]) for i in feats]
    tag_avg = sum(tag_avg)/len(tag_avg)
    return ann_colors[int(tag_avg)]

def render_faithfulness_interface():
    st.header("XSUM with Faithfulness Annotations")

    st.write(f"**# of Annotated summaries:** {len(annotated_data_by_id)}")
    selected_id = str(st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys()))

    st.write("Hallucinations in document summarization.")
    st.write("Hallucination type 0 looks like this.")
    st.write("Hallucination type 1 looks like this.")

    selected_data = annotated_data_by_id[selected_id]
    selected_annotations = pd.DataFrame(selected_data['faithfulness'])

    st.write("**Document:**")
    st.write(selected_data['document'])

    # st.write("**Ground Truth Summary:**")
    # st.write(selected_data['ground_truth_summary'])

    # summarize annotations for each model summary
    # TODO present these in an ordered way
    # TODO add checkboxes for which workers to include???
    for (g_model, g_summary), g_annotations in selected_annotations.groupby(['system', 'summary']):
        # fields in annotations df:
        # bbcid, system, summary, hallucination_type,
        # hallucinated_span_start, hallucinated_span_end, worker_id
        st.write(f'***{g_model}:***')
        st.write(g_annotations[[
            'hallucination_type', 'hallucinated_span_start', 'hallucinated_span_end', 'worker_id'
        ]])
        ann = [[g_summary, set()]]
        for _, r in g_annotations[
            g_annotations['hallucination_type']!=-1
        ].iterrows():
            h_begin = r['hallucinated_span_start']-1
            h_end = r['hallucinated_span_end']-1
            h_type = r['hallucination_type']
            ann = annotation_overlap(ann,
                h_begin,
                h_end,
                # (r['hallucination_type'],r['worker_id']),
                (r['hallucination_type'],),
            )
        ann = annotation_merge(ann)
        annotated_text.annotated_text(
            *annotation_render(ann, annotation_text, annotation_color)
        )

    # TODO combine factuality together with the highlights
    st.write("**Factuality Annotations:**")
    st.table(selected_data["factuality"])

if __name__ == "__main__":
    render_hallucination_interface()
