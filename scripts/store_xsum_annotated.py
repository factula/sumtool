from sumtool.storage import store_model_summaries
from sumtool.xsum_dataset import XsumDataset
from datasets import load_dataset
from collections import defaultdict

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
                model_name = val["system"]
                persisted_summaries_by_model[model_name][bbc_id] = val["summary"]
                persisted_metadata_by_model[model_name][bbc_id] = {
                    "faithfulness_annotations": val["labels"]
                }

        factuality_data = {}
        if len(data["factuality_data"]) > 0:
            for system in data["factuality_data"].keys():
                labels = data["factuality_data"][system]["labels"]
                factuality_data[system] = data["factuality_data"][system]
                factuality_data[system]["mean_worker_factuality_score"] = sum(
                    labels.values()
                ) / len(labels)

        for system, data in factuality_data.items():
            if bbc_id not in persisted_metadata_by_model[system]:
                persisted_metadata_by_model[system][bbc_id] = {}
            persisted_metadata_by_model[system][bbc_id][
                "mean_worker_factuality_score"
            ] = data["mean_worker_factuality_score"]

    for model_name, generated_summaries in persisted_summaries_by_model.items():
        store_model_summaries(
            "xsum",
            model_name,
            persisted_model_config,
            generated_summaries,
            persisted_metadata_by_model[model_name],
        )
