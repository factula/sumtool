import streamlit as st
from datasets import load_dataset
from sumtool.xsum_dataset import XsumDataset


def prepare_data_for_view(data):
    factuality_data = {}
    for system in data["factuality_data"].keys():
        labels = data["factuality_data"][system]["labels"]
        factuality_data[system] = data["factuality_data"][system]
        factuality_data[system]["mean_worker_factuality_score"] = (
            sum(labels.values()) / len(labels)
        )

    flat_faithfulness_data = []
    for val in data["faithfulness_data"].values():
        for labels in val["labels"].values():
            flattened = {
                "system": val["system"],
                "summary": val["summary"],
            }
            for key, value in labels.items():
                flattened[key] = value
            flat_faithfulness_data.append(flattened)

    return {
        "document": data["document"],
        "factuality_data": factuality_data,
        "faithfulness_data": flat_faithfulness_data
    }

@st.experimental_memo
def load_annotated_data_by_id():
    dataset = XsumDataset(
        load_dataset("xsum")["test"],
        factuality_data=load_dataset("xsum_factuality")["train"],
        faithfulness_data=load_dataset("xsum_factuality", "xsum_faithfulness")["train"]
    )
    view_dataset = {
        k: prepare_data_for_view(v)
        for k, v in dataset.data_by_id.items()
        if len(v["faithfulness_data"]) > 0
    }
    return view_dataset
