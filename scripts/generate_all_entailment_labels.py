from sumtool.storage import get_summaries, store_summary_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset
from sumtool.xsum_dataset import XsumDataset
from typing import List, Tuple


def load_entailment_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained("madlag/bert-large-uncased-mnli")
    model = AutoModelForSequenceClassification.from_pretrained(
        "madlag/bert-large-uncased-mnli"
    ).to(device)

    return tokenizer, model


def construct_entailment_data_for_model(
    xsum_examples_by_id: dict, model_info: Tuple[str, str]
) -> List[dict]:
    """
    Merges the main xsum dataset with the model and human generated summaries stored
    in JSON into a list of data consumable by a dataloader.

    Args:
        xsum_examples_by_id: xsum dataset keyed by bbcid
        model_info: tuple of (model name, model hash) used to index into stored summaries

    Returns:
        list of dict of bbcid, xsum document and the summary produced by the model.
    """

    model_name, model_hash = model_info
    stored_summaries = get_summaries("xsum", model_name)[model_hash]

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

    generated_summary_lookup = {
        "BERTS2S": "2221cf6c3bad26c64a12417282dabf3c",
        "TranS2S": "adaab4ed6990eeae0ea8e98c78a722fe",
        "PtGen": "5641d358ff7a45cc65592938f1020675",
        "TConvS2S": "dfadf4c1e1847488969892cc58be06c1",
        "Gold": "dfd16d395e88b38829ff2d5b98743404",
    }

    for model_name, model_hash in generated_summary_lookup.items():
        print(f"Started: {model_name} summaries enailment probs")
        data_to_load = construct_entailment_data_for_model(
            xsum_test_by_id, (model_name, model_hash)
        )
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

            batch_entailment_probs = (
                model(**batch_tokenized)["logits"]
                .softmax(dim=1)
                .cpu()
                .detach()
                .numpy()
                .tolist()
            )

            batch_entailment_metric = [
                dict(zip(["entails_prob", "neutral_prob", "contradicts_prob"], probs))
                for probs in batch_entailment_probs
            ]
            entailment_metrics.extend(list(zip(batch["id"], batch_entailment_metric)))

        store_summary_metrics("xsum", model_name, model_hash, dict(entailment_metrics))
        print(f"Finished: {model_name} summaries enailment probs")
