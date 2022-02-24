import argparse
import torch
import datasets
from typing import List, Tuple
from sumtool.xsum_dataset import XsumDataset
from sumtool.storage import store_model_summaries
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_summarization_model_and_tokenizer() -> Tuple[
    BartForConditionalGeneration, BartTokenizer
]:
    """
    Load summary generation model and move to GPU, if possible.

    Returns:
        (model, tokenizer)
    """
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum")
    model.to(device)

    return model, tokenizer


def generate_summaries(
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    docs_to_summarize: List[str],
) -> str:
    """
    Given a trained summary generation model and appropriate tokenizer,

    1. Tokenize text (and move to device, if possible)
    2. Run inference on model to generate output vocabulary tokens for summary
    3. Decode tokens to a sentence using the tokenizer

    Args:
        model: model to run inference on
        tokenizer: tokenizer corresponding to model
        docs_to_summarize: documents to summarize

    Returns:
        decoded_sentence
    """
    inputs = tokenizer(
        docs_to_summarize,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    input_token_ids = inputs.input_ids.to(device)

    summary_ids = model.generate(
        input_token_ids,
        num_beams=4,
        max_length=150,
        early_stopping=True,
    )

    generated_summaries = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in summary_ids
    ]

    return generated_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run inference on an xsum example using a pre-trained model"
    )

    parser.add_argument(
        "--bbc_ids",
        type=str,
        required=True,
        help="Comma-separated document BBC IDs in the Xsum dataset",
    )

    parser.add_argument(
        "--data_split",
        type=str,
        required=True,
        choices=["train", "test", "validation"],
        help="xsum data split to index into with `data_index`",
    )

    args = parser.parse_args()

    model, tokenizer = load_summarization_model_and_tokenizer()

    xsum_data = XsumDataset(datasets.load_dataset("xsum")[args.data_split])
    selected_data = [xsum_data.data_by_id[x.strip()] for x in args.bbc_ids.split(",")]

    summaries = generate_summaries(
        model,
        tokenizer,
        [x["document"] for x in selected_data]
    )

    for source, gen_summary in zip(selected_data, summaries):
        print("XSUM ID", source["id"])
        print("GOLD STANDARD SUMMARY:", source["true_summary"])
        print("PREDICTED SUMMARY:", gen_summary)

    store_model_summaries(
        "xsum",
        model.config.name_or_path,
        model.config.to_dict(),
        {source["id"]: gen_summary for source, gen_summary in zip(selected_data, summaries)},
    )
