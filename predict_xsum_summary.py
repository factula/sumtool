import argparse
import torch
import datasets


from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run inference on an xsum example using a pre-trained model"
    )

    parser.add_argument(
        '--data_index',
        type=int,
        required=True,
        help='id of the example in the xsum dataset'
    )

    parser.add_argument(
        '--data_split',
        type=str,
        required=True,
        choices=['train', 'test', 'validation'],
        help='xsum data split to index into with `data_index`'
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # configure model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")
    model.to(device)
    max_tokens = 1024

    # load xsum data
    xsum_data = datasets.load_dataset("xsum")

    print(xsum_data[args.data_split][args.data_index])
