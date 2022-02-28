import threading
import streamlit as st

from datasets import load_dataset
from os.path import exists
from enum import Enum

from sumtool.ngram import NgramLookup, preprocess


class Case(Enum):
    no_query_given = 0
    unk_in_query = 1
    match_not_found = 2
    match_found = 3


@st.experimental_memo
def load_xsum_dataset():
    x_sum_dataset = load_dataset("xsum")
    x_sum_dataset = x_sum_dataset["train"]

    return x_sum_dataset


def build_ngram_lookup(
    x_sum_dataset, vocabs_path, ngram_path, max_vocab_size, min_n, max_n, save_flag
):
    # build vocab, ngram_dicts
    # Preprocess documents
    print("Preprocessing documents...")
    x_sum_dataset = x_sum_dataset.map(
        lambda example: {"pp_document": preprocess(example["document"])}, num_proc=5
    )

    # ngram lookup
    ngram_lookup = NgramLookup(documents=x_sum_dataset["pp_document"])

    # build dictionary with multithreading
    p = threading.Thread(
        target=ngram_lookup.build_dictionary,
        args=(vocabs_path, max_vocab_size, save_flag),
    )
    p.start()
    p.join()

    # build ngram dictionary with multithreading
    p2 = threading.Thread(
        target=ngram_lookup.build_ngram_dictionary,
        args=(ngram_path, min_n, max_n, save_flag),
    )
    p2.start()
    p2.join()

    return ngram_lookup


def load_ngram_lookup(vocabs_path, ngram_path, max_vocab_size, min_n, max_n):
    # vocab, ngram_dicts are already built
    assert exists(vocabs_path), "Build vocab first"
    for n in range(min_n, max_n + 1):
        assert exists(ngram_path % n), "Build %d-gram dictionary first"

    # ngram lookup
    ngram_lookup = NgramLookup(documents=None)

    # build dictionary
    ngram_lookup.build_dictionary(
        vocabs_path=vocabs_path, max_vocab_size=max_vocab_size
    )

    # build ngram dictionary
    ngram_lookup.build_ngram_dictionary(ngram_path=ngram_path, min_n=min_n, max_n=max_n)

    return ngram_lookup
