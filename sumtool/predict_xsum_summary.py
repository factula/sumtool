import argparse
import torch
import datasets
from typing import Tuple
from sumtool.xsum_dataset import XsumDataset
from transformers import BartTokenizer, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessor, LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList
from scipy.stats import entropy


def load_summarization_model_and_tokenizer(
    device: torch.cuda.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[BartForConditionalGeneration, BartTokenizer]:
    """
    Load summary generation model and move to GPU, if possible.

    Optional args:
        device: cpu or gpu

    Returns:
        (model, tokenizer)
    """
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum")
    model.to(device)

    return model, tokenizer


def calculate_entropy(seq_scores):
    return torch.vstack([
        torch.tensor(entropy(torch.nn.functional.softmax(
            scores
        ), axis=1))
        for scores in seq_scores
    ])


def predict_summary(
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    text_to_summarize: str,
    analyze_prediction: bool = False,
    num_beams: int = 4,
    device: torch.cuda.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    Optional args:
        analyze_prediction: whether prediction should be analyzed
        device: cpu or gpu

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

    model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        max_length=150,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
        num_return_sequences=1
    )

    predicted_summary = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in model_output.sequences
    ]

    if not analyze_prediction:
        return predicted_summary[0]

    else:

        seq_idx = 0
        prediction_analysis = []
        input_set = input_token_ids.view(-1).tolist()

        if num_beams == 1:
            for idx, score in enumerate(model_output.scores):
                probs = torch.nn.functional.softmax(
                    model_output.scores[idx],
                    dim=1
                ).view(-1)
                output_token_id = model_output.sequences[seq_idx][idx + 1]
                prediction_analysis.append((
                    output_token_id.item(),
                    tokenizer.decode(output_token_id),
                    entropy(probs),
                    output_token_id in input_set
                ))
        else:
            for idx, output_token_id in enumerate(model_output.sequences[seq_idx][1:]):
                beam_idx = model_output.beam_indices[seq_idx][idx]
                beam_probs = torch.exp(model_output.scores[idx][beam_idx])
                beam_token_id = output_token_id

                alternatives = []
                top_probs = torch.topk(beam_probs, k=3)
                for i, v in zip(top_probs.indices, top_probs.values):
                    alternatives.append(f"{tokenizer.decode(i)} ({i.item()})")
                    alternatives.append(v.item())

                prediction_analysis.append((
                    f"{tokenizer.decode(output_token_id)} ({beam_token_id.item()})",
                    beam_probs[output_token_id].item(),
                    beam_idx.item(),
                    entropy(beam_probs).round(3),
                    output_token_id in input_set,
                    *alternatives
                ))

        return predicted_summary[0], prediction_analysis, model_output.sequences_scores.item()


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
    xsum_example = xsum_train_data.query_by_bbc_id(args.bbc_id)

    summary = predict_summary(model, tokenizer, xsum_example["document"])

    print("GOLD STANDARD SUMMARY:", xsum_example["true_summary"])
    print("PREDICTED SUMMARY:", summary)
