from sumtool.storage import store_model_summaries
from sumtool.xsum_dataset import XsumDataset
from datasets import load_dataset
from collections import defaultdict


def map_data_for_storage(data):
    factuality_data = {}
    for system in data["factuality_data"].keys():
        labels = data["factuality_data"][system]["labels"]
        factuality_data[system] = data["factuality_data"][system]
        factuality_data[system]["mean_worker_factuality_score"] = sum(
            labels.values()
        ) / len(labels)

    flat_faithfulness_data = []
    for val in data["faithfulness_data"].values():
        for label in val["labels"]:
            flattened = {
                "system": val["system"],
                "summary": val["summary"],
            }
            for label_key, label_value in label.items():
                flattened[label_key] = label_value
            flat_faithfulness_data.append(flattened)

    return {
        "document": data["document"],
        "factuality_data": factuality_data,
        "faithfulness_data": flat_faithfulness_data,
    }


if __name__ == "__main__":
    dataset = XsumDataset(
        load_dataset("xsum")["test"],
        factuality_data=load_dataset("xsum_factuality")["train"],
        faithfulness_data=load_dataset("xsum_factuality", "xsum_faithfulness")["train"],
    )

    persisted_summaries_by_model = defaultdict(lambda: {})
    persisted_metadata_by_model = defaultdict(lambda: {})
    persisted_model_config = {
        "Paper": "On Faithfulness and Factuality in Abstractive Summarization (2020)",
        "Authors": "Joshua Maynez, Shashi Narayan, Bernd Bohnet, Ryan McDonald",
        "Link": "https://arxiv.org/pdf/2005.00661",
    }

    for bbc_id, data in dataset.data_by_id.items():
        if len(data["faithfulness_data"]) > 0:
            for val in data["faithfulness_data"].values():
                for label in val["labels"]:
                    model_name = val["system"]
                    flattened = {
                        "system": val["system"],
                        "summary": val["summary"],
                    }
                    persisted_summaries_by_model[model_name][bbc_id] = val["summary"]

                    # TODO add faithfulness annotations in metadata

        factuality_data = {}
        if len(data["factuality_data"]) > 0:
            for system in data["factuality_data"].keys():
                labels = data["factuality_data"][system]["labels"]
                factuality_data[system] = data["factuality_data"][system]
                factuality_data[system]["mean_worker_factuality_score"] = sum(
                    labels.values()
                ) / len(labels)

        for system, data in factuality_data.items():
            persisted_metadata_by_model[system][bbc_id] = {
                "mean_worker_factuality_score": data["mean_worker_factuality_score"]
            }

    for model_name, generated_summaries in persisted_summaries_by_model.items():
        store_model_summaries(
            "xsum",
            model_name,
            persisted_model_config,
            generated_summaries,
            persisted_metadata_by_model[model_name],
        )
