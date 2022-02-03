import argparse
import torch
import datasets


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

    # configure model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")
    model.to(device)

    max_num_input_tokens = 1024
    max_num_output_tokens = 150

    # load xsum data
    xsum_data = datasets.load_dataset("xsum")
    xsum_example = xsum_data[args.data_split][args.data_index]

    inputs = tokenizer(
        [xsum_example["document"]],
        max_length=max_num_input_tokens,
        truncation=True,
        return_tensors="pt",
    )
    input_token_ids = inputs.input_ids.to(device)

    summary_ids = model.generate(
        input_token_ids,
        num_beams=4,
        max_length=max_num_output_tokens,
        early_stopping=True,
    )

    predicted_summary = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in summary_ids
    ]

    print("GOLD STANDARD SUMMARY:", xsum_example["summary"])
    print("PREDICTED SUMMARY:", predicted_summary[0])
