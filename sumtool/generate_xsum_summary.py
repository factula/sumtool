import argparse
import torch
import datasets
from typing import List, Tuple
from sumtool.utils import entropy
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
    num_beams: int = 4,
    return_generation_metadata: bool = False
):
    """
    Given a trained summary generation model and appropriate tokenizer,

    1. Tokenize text (and move to device, if possible)
    2. Run inference on model to generate output vocabulary tokens for summary
    3. Decode tokens to a sentence using the tokenizer

    Args:
        model: model to run inference on
        tokenizer: tokenizer corresponding to model
        docs_to_summarize: documents to summarize
        num_beams: number of beams for beam search
        return_generation_metadata: whether generation metadata should be returned

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

    model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        max_length=150,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_summaries = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in model_output.sequences
    ]

    if not return_generation_metadata:
        return generated_summaries
    else:
        token_metadata = []
        input_set = input_token_ids.view(-1).tolist()
        for seq_idx in range(model_output.sequences.shape[0]):
            seq_metadata = []
            token_metadata.append(seq_metadata)
            for idx, output_token_id in enumerate(model_output.sequences[seq_idx][1:]):
                beam_idx = model_output.beam_indices[seq_idx][idx]
                selected_beam_probs = torch.exp(model_output.scores[idx][beam_idx])

                beam_top_alternatives = []
                top_probs = torch.topk(selected_beam_probs, k=3)
                for i, v in zip(top_probs.indices, top_probs.values):
                    beam_top_alternatives.append({
                        "token": tokenizer.decode(i),
                        "token_id": i.item(),
                        "beam_token_prob": v.item()
                    })

                seq_metadata.append({
                    "token_id": output_token_id,
                    "token": tokenizer.decode(output_token_id),
                    "entropy": entropy(selected_beam_probs),
                    "beam_token_prob": selected_beam_probs[output_token_id].item(),
                    "beam_idx": beam_idx.item(),
                    "beam_top_probs": beam_top_alternatives,
                    "token_in_input": output_token_id in input_set,
                })

        return generated_summaries, token_metadata


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

    summaries, generation_metadata = generate_summaries(
        model,
        tokenizer,
        [x["document"] for x in selected_data],
        num_beams=4,
        return_generation_metadata=True
    )

    summary_metadata = {}

    for source, gen_summary, seq_metadata in zip(selected_data, summaries, generation_metadata):
        print("XSUM ID", source["id"])
        print("GOLD STANDARD SUMMARY:", source["true_summary"])
        print("PREDICTED SUMMARY:", gen_summary)

        tokens_with_entropy = []
        for token_metadata in seq_metadata:
            tokens_with_entropy.append((
                token_metadata["token"],
                token_metadata["entropy"]
            ))

        summary_metadata[source["id"]] = {
            "tokens_with_entropy": tokens_with_entropy
        }

    store_model_summaries(
        "xsum",
        model.config.name_or_path,
        model.config.to_dict(),
        {
            source["id"]: gen_summary
            for source, gen_summary in zip(selected_data, summaries)
        },
        summary_metadata
    )
