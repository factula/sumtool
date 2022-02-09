import streamlit as st
import annotated_text
import pandas as pd
from backend.viz_data_loader import load_annotated_data_by_id

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

def annotation_render(ann, func_text, func_color, max_count=3, render_presence=False, render_type=False):
    return [
        (e[0]) if (len(e[1])==0) else (
            e[0],
            func_text(e[1]),
            func_color(e[1], render_presence, render_type, max_count=max_count)
        )
        for e in ann
    ]

def annotation_text(feats):
    # Assume from the XSum data that there are up to 3 workers, with tags of 0/1
    tag_avg = [int(i[0]) for i in feats]
    tag_avg = sum(tag_avg)/len(tag_avg)
    return f'{str(tag_avg)[:3]}:{len(feats)}'

def annotation_color(feats, render_presence, render_type, max_count=3):
    # Assume from the XSum data that there are up to 3 workers, with tags of 0/1
    def colorcombine(a, b, frac_a, render_type):
        if render_type:
            return [(frac_a*a[i]+(1-frac_a)*b[i]) for i in range(len(a))]
        else:
            return a if (round(frac_a)==1) else b
    def cssify(t):
        if render_presence:
            return f'rgba({int(t[0])}, {int(t[1])}, {int(t[2])}, {t[3]})'
        else:
            return f'rgba({int(t[0])}, {int(t[1])}, {int(t[2])})'
    ann_colors = {
        1: (26, 133, 255),
        0: (212, 17, 89),
    }
    tag_avg = [int(i[0]) for i in feats]
    tag_avg = sum(tag_avg)/len(tag_avg)
    return cssify((
        *colorcombine(ann_colors[1], ann_colors[0], tag_avg, render_type),
        len(feats)/max_count
    ))

def render_faithfulness_interface():
    st.header("XSUM with Faithfulness Annotations")

    st.write(f"**# of Annotated summaries:** {len(annotated_data_by_id)}")
    selected_id = str(st.selectbox("Select entry by bbcid", options=annotated_data_by_id.keys()))

    render_ann_presence = st.checkbox('Distinguish annotation presence', value=False)
    render_ann_halltype = st.checkbox('Distinguish annotation type', value=False)

    annotated_text.annotated_text(*annotation_render(
        [
            ['Hallucination type 0 (intrinsic?).', {(0,'1'),(0,'2'),(0,'3')}],
            ['When only some annotators notice it.', {(0,'1'),(0,'2')}],
        ],
        annotation_text, annotation_color, max_count=3,
        render_presence=render_ann_presence, render_type=render_ann_halltype,
    ))
    annotated_text.annotated_text(*annotation_render(
        [
            ['Hallucination type 1 (extrinsic?).', {(1,'1'),(1,'2'),(1,'3')}],
            ['When only some annotators notice it.', {(1,'1'),(1,'2')}],
        ],
        annotation_text, annotation_color, max_count=3,
        render_presence=render_ann_presence, render_type=render_ann_halltype,
    ))
    annotated_text.annotated_text(*annotation_render(
        [
            ['Hallucination type mostly 0.', {(0,'1'),(0,'2'),(0,'3'),(0,'4'),(1,'5'),(1,'6')}],
            ['When only some annotators notice it.', {(0,'1'),(0,'2'),(1,'5')}],
        ],
        annotation_text, annotation_color, max_count=6,
        render_presence=render_ann_presence, render_type=render_ann_halltype,
    ))
    annotated_text.annotated_text(*annotation_render(
        [
            ['Hallucination type mostly 1.', {(1,'1'),(1,'2'),(1,'3'),(1,'4'),(0,'5'),(0,'6')}],
            ['When only some annotators notice it.', {(1,'1'),(1,'2'),(0,'5')}],
        ],
        annotation_text, annotation_color, max_count=6,
        render_presence=render_ann_presence, render_type=render_ann_halltype,
    ))

    st.write('---')

    selected_data = annotated_data_by_id[selected_id]
    selected_annotations = pd.DataFrame(selected_data['faithfulness'])
    selected_factuality = pd.DataFrame(selected_data['factuality'])

    # summarize annotations for each model summary
    # TODO present these in an ordered way
    # TODO add checkboxes for which workers to include???
    for (g_model, g_summary), g_annotations in selected_annotations.groupby(['system', 'summary']):
        # fields in annotations df:
        # bbcid, system, summary, hallucination_type,
        # hallucinated_span_start, hallucinated_span_end, worker_id
        st.write(f'**{g_model}:**')
        # if g_model!='Gold':
        #     g_factuality = selected_factuality[
        #         selected_factuality['system']==g_model
        #     ].iloc[0]['mean_worker_factuality_score']
        #     st.write(f'mean_worker_factuality_score={str(g_factuality)}')
        # st.write(g_annotations)
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
                (r['hallucination_type'],r['worker_id']),
                # (r['hallucination_type'],),
            )
        ann = annotation_merge(ann)
        annotated_text.annotated_text(*annotation_render(
            ann, annotation_text, annotation_color, max_count=3,
            render_presence=render_ann_presence, render_type=render_ann_halltype,
        ))
        st.write('')

    st.write("**Document:**")
    st.write(selected_data['document'])

    # st.write("**Ground Truth Summary:**")
    # st.write(selected_data['ground_truth_summary'])

    st.write("**Faithfulness Annotation source data:**")
    st.write(selected_annotations)

if __name__ == "__main__":
    render_faithfulness_interface()
