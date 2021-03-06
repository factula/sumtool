# -*- coding: utf-8 -*-
"""2022-02-03 CS6741 summary + QA test

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ubo3Q_cIN3z5C3Tm0xvZ1QzGmB7IRiCY
"""

from typing import Dict, List

import argparse
import functools
import pandas as pd
import tqdm

from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import seaborn as sns
import matplotlib.pyplot as plt


def get_xsum_datapoints(n: int = 1000, shuffle: bool = False) -> Dict[str, List[str]]:
    """Loads xsum data and computes auto_summary field using summarizer.

    Args:
        n (int): number of datapoints to return
        shuffle: (bool) whether to shuffle dataset

    Returns:
        {
            'document': List[str] of original documents from xsum data
            'summary': List[str] of ground-truth summaries from xsum data
            'auto_summary': List[str] of machine-written summaries for documents
        }
    """
    summarizer = pipeline(
        "summarization", model="t5-base", tokenizer="t5-base", device=0
    )
    xsum = load_dataset("xsum")
    xsum_train = xsum["train"]
    if shuffle:
        xsum_train = xsum_train.shuffle(seed=42)
    xsum_train = xsum_train[:n]
    summaries = summarizer(xsum_train["document"])
    xsum_train["auto_summary"] = [r["summary_text"] for r in summaries]
    return xsum_train


# Function for getting entailment label
_NLI = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}


def get_entailment_label_from_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    premise: str,
    hypothesis: str,
) -> str:
    """Gets NLI label for (premise, hypothesis) pair: one of ['Entailment', 'Neutral', 'Contradiction'].

    Args:
        model (AutoModelForSequenceClassification): NLI model to use (must respect the labeling
            convention: 0-entailment, 1-neutral, 2-contradiction)
        tokenizer (AutoTokenizer): tokenizer for provided model
        premise (str): the premise for entailment
        hypothesis (str): the hypothesis for entailment

    Returns:
        str: label, one of ['Entailment', 'Neutral', 'Contradiction']
    """
    tokens = tokenizer(
        premise, hypothesis, return_tensors="pt", padding=True, truncation=True
    )
    outputs = model(**tokens)
    label_idx = outputs["logits"].squeeze().argmax().item()
    return _NLI[label_idx]


def plot_nli_data(xsum_train: Dict[str, List[str]]):
    """Creates a plot showing the frequencies of each NLI label in human-written and machine-written summaries.

    Args:
        xsum_train: a Dict[str, List[str]] with the following structure
            {
                'document': List[str] of original documents from xsum data
                'summary': List[str] of ground-truth summaries from xsum data
                'auto_summary': List[str] of machine-written summaries for documents
                'nli_label': List[int] of labels for ground-truth summaries
                'auto_nli_label': List[int] of labels for machine-written summaries
            }
    """
    #
    # Create dataframe with all this stuff
    #
    rows = []
    n = len(xsum_train["document"])
    for i in range(n):
        rows.append(
            [
                xsum_train["document"][i],
                xsum_train["summary"][i],
                xsum_train["nli_label"][i],
                "ground_truth",
            ]
        )
        rows.append(
            [
                xsum_train["document"][i],
                xsum_train["auto_summary"][i],
                xsum_train["auto_nli_label"][i],
                "auto_generated",
            ]
        )
    df = pd.DataFrame(
        rows, columns=["document", "summary", "nli_label", "autogenerated"]
    )
    df.head()
    #
    # Create plot of NLI labels
    #
    sns.set(style="white")
    f = plt.figure(figsize=(14, 6))
    ax = f.add_subplot(1, 1, 1)
    sns.histplot(
        data=df,
        ax=ax,
        stat="count",
        multiple="stack",
        y="nli_label",
        kde=False,
        palette="pastel",  # pastel, muted, bright... https://seaborn.pydata.org/tutorial/color_palettes.html
        hue="autogenerated",
        element="bars",
        legend=True,
        bins=3,
    )
    ax.set_title("NLI Labels for document-summary pairs")
    ax.set_xlabel("Number of labels")
    ax.set_ylabel("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summaries and evaluate them using NLI model."
    )
    parser.add_argument(
        "--n", type=int, default=100, help="number of examples to summarize"
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="whether to shuffle training data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Get data
    xsum_train = get_xsum_datapoints(n=args.n, shuffle=args.shuffle)

    # Load entailment models
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        "typeform/distilbert-base-uncased-mnli"
    )
    nli_tokenizer = AutoTokenizer.from_pretrained(
        "typeform/distilbert-base-uncased-mnli"
    )

    get_entailment_label = functools.partial(
        get_entailment_label_from_model, nli_model, nli_tokenizer
    )
    # Get entailment labels for 1000 human-written summaries
    xsum_train["nli_label"] = list(
        tqdm.tqdm(
            map(
                lambda r: get_entailment_label(*r),
                zip(xsum_train["document"], xsum_train["summary"]),
            ),
            total=args.n,
        )
    )
    # Get entailment labels for 1000 machine-written summaries
    xsum_train["auto_nli_label"] = list(
        tqdm.tqdm(
            map(
                lambda r: get_entailment_label(*r),
                zip(xsum_train["document"], xsum_train["auto_summary"]),
            ),
            total=args.n,
        )
    )

    plot_nli_data(xsum_train)


if __name__ == "__main__":
    main()
