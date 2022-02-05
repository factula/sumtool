from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import csv


class XsumDataset(Dataset):
    def __init__(self, xsum_data, factuality_data=None, faithfulness_data=None):
        self.xsum_data = xsum_data
        self.factuality_data = factuality_data
        self.faithfulness_data = faithfulness_data

        self.dataset = list(self._align_data().items())

    def _align_data(self):
        """Aligns data in xsum, factuality, and faithfulness datasets:
        Returns:
            dataset:  Dict(id, {
                'document': original document,
                'true_summary': true summary,
                'factuality_data': Dict(system name, {
                    'summary': system summary,
                    'labels': Dict(worker_id: is_factual)
                },
                'faithfulness_data':[
                    Dict(system name: {
                        'summary': system summary,
                        'hallucinations': Dict(worker_id, {
                            'hallucination_type': hallucination type,
                            'hallucinated_span': hallucinated span,
                            'hallucinated_span_start': hallucinated span start,
                            'hallucinated_span_end': hallucinated span end
                        })
                    }
                ]
            }
        )

        """

        dataset = {}

        for data in self.xsum_data:
            dataset[str(data["id"])] = {
                "document": data["document"],
                "true_summary": data["summary"],
                "factuality_data": {},
                "faithfulness_data": {},
            }

        if self.factuality_data is not None:
            for data in self.factuality_data:
                if str(data["bbcid"]) not in dataset:
                    dataset[str(data["bbcid"])] = {
                        "document": None,
                        "true_summary": None,
                        "factuality_data": {},
                        "faithfulness_data": {},
                    }
                if data["system"] in dataset[str(data["bbcid"])]["factuality_data"]:
                    dataset[str(data["bbcid"])]["factuality_data"][data["system"]][
                        "labels"
                    ][str(data["worker_id"])] = data["is_factual"]
                else:
                    dataset[str(data["bbcid"])]["factuality_data"][data["system"]] = {
                            "summary": data["summary"],
                            "labels": {data["worker_id"]: data["is_factual"]},
                        }
                    
        if self.faithfulness_data is not None:
            for data in self.faithfulness_data:
                if str(data["bbcid"]) not in dataset:
                    dataset[str(data["bbcid"])] = {
                        "document": None,
                        "true_summary": None,
                        "factuality_data": {},
                        "faithfulness_data": {},
                    }
                if data["system"] in dataset[str(data["bbcid"])]["faithfulness_data"]:
                    dataset[str(data["bbcid"])]["faithfulness_data"][data["system"]][str(data["worker_id"])] = {
                        "hallucination_type": data["hallucination_type"],
                        "hallucinated_span": data["summary"][
                            data["hallucinated_span_start"] : data[
                                "hallucinated_span_end"
                            ]
                        ],
                        "hallucinated_span_start": data["hallucinated_span_start"],
                        "hallucinated_span_end": data["hallucinated_span_end"],
                    }
                else:
                    dataset[str(data["bbcid"])]["faithfulness_data"][data["system"]] = {
                        str(data["worker_id"]): {
                            "hallucination_type": data["hallucination_type"],
                            "hallucinated_span": data["summary"][
                                data["hallucinated_span_start"] : data[
                                    "hallucinated_span_end"
                                ]
                            ],
                            "hallucinated_span_start": data["hallucinated_span_start"],
                            "hallucinated_span_end": data["hallucinated_span_end"],
                        }
                    }
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


data_xsum = load_dataset("xsum")  # document (string), summary (string), id (string)
data_xsum_factuality = load_dataset(
    "xsum_factuality"
)  # bbcid (int), system (string), summary (string), is_factual (class label), worker_id (string)
data_xsum_faithfulness = load_dataset(
    "xsum_factuality", "xsum_faithfulness"
)  # bbcid,system,summary,hallucination_type,hallucinated_span_start,hallucinated_span_end,worker_id

train_data = XsumDataset(
    data_xsum["train"], data_xsum_factuality["train"], data_xsum_faithfulness["train"]
)
val_data = XsumDataset(data_xsum["validation"])
test_data = XsumDataset(data_xsum["test"])

train_dl = DataLoader(train_data, batch_size=4, shuffle=False)
val_dl = DataLoader(val_data, batch_size=4, shuffle=False)
test_dl = DataLoader(test_data, batch_size=4, shuffle=False)

for item in train_dl:
    print(item)
    break


for item in val_dl:
    print(item)
    break

# TODO
# issues I see:
# spans of hallucinations seem off by one character --> one-indexing vs zero-indexing?
