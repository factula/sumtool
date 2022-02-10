import annotated_text
import streamlit as st


def annotation_overlap(ann, n_start, n_end, n_details):
    if n_start > n_end:
        raise ValueError()
    # find indexes of beginning and end chunk presence
    prevstart_i = 0
    annstart_i = 0
    while annstart_i < len(ann) and prevstart_i + len(ann[annstart_i][0]) <= n_start:
        prevstart_i += len(ann[annstart_i][0])
        annstart_i += 1
    prevend_i = prevstart_i
    annend_i = annstart_i
    while annend_i < len(ann) and prevend_i + len(ann[annend_i][0]) <= n_end:
        prevend_i += len(ann[annend_i][0])
        annend_i += 1
    # check if we only need to modify one chunk
    if annstart_i >= len(ann):
        pass
    elif annstart_i == annend_i:
        if n_end > 0:
            ann.insert(annstart_i, [ann[annstart_i][0], ann[annstart_i][1].copy()])
            ann.insert(annstart_i, [ann[annstart_i][0], ann[annstart_i][1].copy()])
            ann[annstart_i][0] = ann[annstart_i][0][: n_start - prevstart_i]
            ann[annstart_i + 1][0] = ann[annstart_i + 1][0][
                n_start - prevstart_i : n_end - prevend_i
            ]
            ann[annstart_i + 2][0] = ann[annstart_i + 2][0][n_end - prevend_i :]
            ann[annstart_i + 1][1].add(n_details)
    else:
        # this needs to be done across multiple chunks
        # add to beginning chunk
        if n_start <= prevstart_i:
            ann[annstart_i][1].add(n_details)
        else:
            # split out a new beginning subchunk
            ann.insert(annstart_i, [ann[annstart_i][0], ann[annstart_i][1].copy()])
            ann[annstart_i][0] = ann[annstart_i][0][: n_start - prevstart_i]
            ann[annstart_i + 1][0] = ann[annstart_i + 1][0][n_start - prevstart_i :]
            ann[annstart_i + 1][1].add(n_details)
            annstart_i += 1
            annend_i += 1
        # add to intermediate chunk(s)
        for i in range(annstart_i + 1, annend_i - 1):
            ann[i][1].add(n_details)
        # add to end chunk
        if annend_i == len(ann) or n_end == prevend_i + len(ann[annend_i - 1][0]):
            if annend_i - 1 != annstart_i:
                ann[annend_i - 1][1].add(n_details)
        else:
            # split out a new end subchunk
            ann.insert(annend_i, [ann[annend_i][0], ann[annend_i][1].copy()])
            ann[annend_i][0] = ann[annend_i][0][: n_end - prevend_i]
            ann[annend_i + 1][0] = ann[annend_i + 1][0][n_end - prevend_i :]
            if annend_i - 1 != annstart_i:
                ann[annend_i - 1][1].add(n_details)
            ann[annend_i][1].add(n_details)
    # clean out the chunks
    ann = [e for e in ann if len(e[0]) != 0]
    return ann


# Combine sequential chunks of annotation list that have identical tagging
def annotation_merge(ann):
    if len(ann) == 0:
        return ann
    # keep appending as long as next piece is identical to last piece
    new_ann = [ann[0]]
    for a in ann[1:]:
        if a[1] == new_ann[-1][1]:
            new_ann[-1][0] += a[0]
        else:
            new_ann.append(a)
    return new_ann


def annotation_render(
    ann, func_text, func_color, max_count=3, render_presence=False, render_type=False
):
    return [
        (e[0])
        if (len(e[1]) == 0)
        else (
            e[0],
            func_text(e[1]),
            func_color(e[1], render_presence, render_type, max_count=max_count),
        )
        for e in ann
    ]


def annotation_text(feats):
    # Assume from the XSum data that there are up to 3 workers, with tags of 0/1
    tag_avg = [int(i[0]) for i in feats]
    tag_avg = sum(tag_avg) / len(tag_avg)
    return f"{str(tag_avg)[:3]}:{len(feats)}"


def annotation_color(feats, render_presence, render_type, max_count=3):
    # Assume from the XSum data that there are up to 3 workers, with tags of 0/1
    def colorcombine(a, b, frac_a, render_type):
        if render_type:
            return [(frac_a * a[i] + (1 - frac_a) * b[i]) for i in range(len(a))]
        else:
            return a if (round(frac_a) == 1) else b

    def cssify(t):
        if render_presence:
            return f"rgba({int(t[0])}, {int(t[1])}, {int(t[2])}, {t[3]})"
        else:
            return f"rgba({int(t[0])}, {int(t[1])}, {int(t[2])})"

    ann_colors = {
        1: (26, 133, 255),
        0: (212, 17, 89),
    }
    tag_avg = [int(i[0]) for i in feats]
    tag_avg = sum(tag_avg) / len(tag_avg)
    return cssify(
        (
            *colorcombine(ann_colors[1], ann_colors[0], tag_avg, render_type),
            len(feats) / max_count,
        )
    )


def render_faithfulness_annotation_legend(render_ann_presence, render_ann_halltype):
    if not render_ann_halltype and not render_ann_presence:
        annotated_text.annotated_text(
            *annotation_render(
                [
                    [
                        "Hallucination type 0 (intrinsic).",
                        {(0, "1"), (0, "2"), (0, "3")},
                    ],
                    [
                        "Hallucination type 1 (extrinsic).",
                        {(1, "1"), (1, "2"), (1, "3")},
                    ],
                ],
                annotation_text,
                annotation_color,
                max_count=3,
                render_presence=render_ann_presence,
                render_type=render_ann_halltype,
            )
        )
        st.write(
            """
        If annotators disagree, the majority annotation is chosen.
        """
        )
    elif render_ann_halltype and not render_ann_presence:
        annotated_text.annotated_text(
            *annotation_render(
                [
                    [
                        "Hallucination type 0 (intrinsic).",
                        {(0, "1"), (0, "2"), (0, "3")},
                    ],
                    [
                        "Hallucination type mostly 0.",
                        {(0, "1"), (0, "2"), (0, "3"), (0, "4"), (1, "5"), (1, "6")},
                    ],
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
                        "Hallucination type 1 (extrinsic).",
                        {(1, "1"), (1, "2"), (1, "3")},
                    ],
                    [
                        "Hallucination type mostly 1.",
                        {(1, "1"), (1, "2"), (1, "3"), (1, "4"), (0, "5"), (0, "6")},
                    ],
                ],
                annotation_text,
                annotation_color,
                max_count=3,
                render_presence=render_ann_presence,
                render_type=render_ann_halltype,
            )
        )
        st.write(
            """
        If annotators disagree, the annotation color is adjusted accordingly.
        """  # noqa
        )
    elif not render_ann_halltype and render_ann_presence:
        annotated_text.annotated_text(
            *annotation_render(
                [
                    [
                        "Hallucination type 0 (intrinsic).",
                        {(0, "1"), (0, "2"), (0, "3")},
                    ],
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
                    [
                        "Hallucination type 1 (extrinsic).",
                        {(1, "1"), (1, "2"), (1, "3")},
                    ],
                    ["When only some annotators notice it.", {(1, "1"), (1, "2")}],
                ],
                annotation_text,
                annotation_color,
                max_count=3,
                render_presence=render_ann_presence,
                render_type=render_ann_halltype,
            )
        )
        st.write(
            """
        If annotators disagree, the majority annotation is chosen.  
        If the annotation is partial (out of 3 workers), the annotation color is adjusted accordingly.  
        """  # noqa
        )
    else:
        annotated_text.annotated_text(
            *annotation_render(
                [
                    [
                        "Hallucination type 0 (intrinsic).",
                        {(0, "1"), (0, "2"), (0, "3")},
                    ],
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
                    [
                        "Hallucination type 1 (extrinsic).",
                        {(1, "1"), (1, "2"), (1, "3")},
                    ],
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
        st.write(
            """
        If annotators disagree, the annotation color is adjusted accordingly.  
        If the annotation is partial (out of 3 workers), the annotation color is adjusted accordingly.  
        """  # noqa
        )

    st.write("---")
