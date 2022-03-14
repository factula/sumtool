from sumtool.storage import get_summaries, store_summary_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset
from sumtool.xsum_dataset import XsumDataset
from typing import List


def load_entailment_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained("madlag/bert-large-uncased-mnli")
    model = AutoModelForSequenceClassification.from_pretrained(
        "madlag/bert-large-uncased-mnli"
    ).to(device)

    return tokenizer, model


def construct_entailment_data_for_model(
    xsum_examples_by_id: dict, model_id: str
) -> List[dict]:
    """
    Merges the main xsum dataset with the model and human generated summaries stored
    in JSON into a list of data consumable by a dataloader.

    Args:
        xsum_examples_by_id: xsum dataset keyed by bbcid
        model_id: model id used to index into stored summaries

    Returns:
        list of dict of bbcid, xsum document and the summary produced by the model.
    """
    stored_summaries = get_summaries("xsum", model_id)

    data_to_load = []

    for bbcid, data in stored_summaries.items():
        example_to_load = {
            "id": bbcid,
            "document": xsum_examples_by_id[bbcid]["document"],
            "summary": data["summary"],
        }
        data_to_load.append(example_to_load)

    return data_to_load


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = load_entailment_model_and_tokenizer(device)

    xsum_test_by_id = XsumDataset(load_dataset("xsum")["test"]).data_by_id

    model_ids = [
        "maynez-berts2s",
        "maynez-gold",
        "maynez-ptgen",
        "maynez-tconvs2s",
        "maynez-trans2s",
    ]

    for model_id in model_ids:
        print(f"Started: {model_id} summaries entailment probs")
        data_to_load = construct_entailment_data_for_model(xsum_test_by_id, model_id)
        data_loader = DataLoader(data_to_load, batch_size=8)

        entailment_metrics = []

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch_tokenized = tokenizer(
                batch["document"],
                batch["summary"],
                padding=True,
                return_tensors="pt",
                truncation="only_first",
            ).to(device)

            with torch.no_grad():
                batch_entailment_probs = (
                    model(**batch_tokenized)["logits"]
                    .softmax(dim=1)
                    .cpu()
                    .numpy()
                    .tolist()
                )

            batch_entailment_metric = [
                dict(zip(["entails_prob", "neutral_prob", "contradicts_prob"], probs))
                for probs in batch_entailment_probs
            ]
            entailment_metrics.extend(list(zip(batch["id"], batch_entailment_metric)))

        store_summary_metrics("xsum", model_id, dict(entailment_metrics))
        print(f"Finished: {model_id} summaries enailment probs")
