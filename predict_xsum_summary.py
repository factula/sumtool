import argparse
import torch
import datasets

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_summarization_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")
    model.to(device)

    return model, tokenizer

def predict_summary(model, tokenizer, text_to_summarize):
    inputs = tokenizer(
        [text_to_summarize],
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
        "--data_index",
        type=int,
        required=True,
        help="id of the example in the xsum dataset",
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

    xsum_data = datasets.load_dataset("xsum")
    xsum_example = xsum_data[args.data_split][args.data_index]

    summary = predict_summary(model, tokenizer, xsum_example["document"])

    print("GOLD STANDARD SUMMARY:", xsum_example["summary"])
    print("PREDICTED SUMMARY:", summary)
