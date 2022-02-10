import argparse
import torch
import datasets
from typing import Tuple
from xsum_dataset import XsumDataset

from transformers import BartTokenizer, BartForConditionalGeneration


def load_summarization_model_and_tokenizer(
    device: torch.cuda.Device,
) -> Tuple[BartForConditionalGeneration, BartTokenizer]:
    """
    Load summary generation model and move to GPU, if possible.

    Args:
        device: cpu or gpu

    Returns:
        (model, tokenizer)
    """
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum")
    model.to(device)

    return model, tokenizer


def predict_summary(
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    text_to_summarize: str,
) -> str:
    """
    Given a trained summary generation model and appropriate tokenizer,

    1. Tokenize text (and move to device, if possible)
    2. Run inference on model to generate output vocabulary tokens for summary
    3. Decode tokens to a sentence using the tokenizer

    Args:
        model: model to run inference on
        tokenizer: tokenizer corresponding to model
        text_to_summarize: document to summarize

    Returns:
        decoded_sentence
    """
    inputs = tokenizer(
        text_to_summarize,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )
    input_token_ids = inputs.input_ids.to(device)

    summary_ids = model.generate(
        input_token_ids,
        num_beams=4,
        max_length=150,
        early_stopping=True,
    )

    predicted_summary = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in summary_ids
    ]

    return predicted_summary[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run inference on an xsum example using a pre-trained model"
    )

    parser.add_argument(
        "--bbc_id",
        type=int,
        required=True,
        help="Document BBC ID in the Xsum dataset",
    )

    parser.add_argument(
        "--data_split",
        type=str,
        required=True,
        choices=["train", "test", "validation"],
        help="xsum data split to index into with `data_index`",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_summarization_model_and_tokenizer(device)

    xsum_train_data = XsumDataset(datasets.load_dataset("xsum")[args.data_split])
    xsum_example = xsum_train_data[args.bbc_id]

    summary = predict_summary(model, tokenizer, xsum_example["document"])

    print("GOLD STANDARD SUMMARY:", xsum_example["true_summary"])
    print("PREDICTED SUMMARY:", summary)
