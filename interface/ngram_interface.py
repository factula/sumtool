import threading
from genericpath import exists
import streamlit as st

# from memory_profiler import profile
from os.path import dirname, realpath, join
from datasets import load_dataset

from sumtool.ngram import NgramLookup, preprocess

# parameters
CURRENT_PATH = dirname(realpath(__file__))  # current file path
VOCABS_PATH = join(CURRENT_PATH, "../sumtool/ngram/cache/vocabs")  # vocab file path
NGRAM_PATH = join(CURRENT_PATH, "../sumtool/ngram/cache/ngram_dict_%d")  # ngram file path
MAX_N = 4  # max n for ngram
MAX_VOCAB_SIZE = 10000  # vocab size
SAVE_FLAG = True  # whether to save vocab, ngram files
NUM_PROC = 5  # # of processes to use for preprocessing


def load_xsum_dataset():
    x_sum_dataset = load_dataset("xsum")
    x_sum_dataset = x_sum_dataset["train"]

    return x_sum_dataset


def build_ngram_lookup(x_sum_dataset):
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
        args=(VOCABS_PATH, MAX_VOCAB_SIZE, SAVE_FLAG),
    )
    p.start()
    p.join()

    # build ngram dictionary with multithreading
    p2 = threading.Thread(
        target=ngram_lookup.build_ngram_dictionary, args=(NGRAM_PATH, MAX_N, SAVE_FLAG)
    )
    p2.start()
    p2.join()

    # ngram lookup
    ngram_lookup = NgramLookup(documents=x_sum_dataset)

    # build dictionary
    ngram_lookup.build_dictionary(
        vocabs_path=VOCABS_PATH, max_vocab_size=MAX_VOCAB_SIZE
    )

    # build ngram dictionary
    ngram_lookup.build_ngram_dictionary(ngram_path=NGRAM_PATH, max_n=MAX_N)

    return ngram_lookup


def load_ngram_lookup():
    # vocab, ngram_dicts are already built
    assert exists(VOCABS_PATH), "Build vocab first"
    for n in range(1, MAX_N + 1):
        assert exists(NGRAM_PATH % n), "Build %d-gram dictionary first"

    # ngram lookup
    ngram_lookup = NgramLookup(documents=None)

    # build dictionary
    ngram_lookup.build_dictionary(
        vocabs_path=VOCABS_PATH, max_vocab_size=MAX_VOCAB_SIZE
    )

    # build ngram dictionary
    ngram_lookup.build_ngram_dictionary(ngram_path=NGRAM_PATH, max_n=MAX_N)

    return ngram_lookup


@st.experimental_memo
def cache_load_xsum_dataset():
    return load_xsum_dataset()


# @st.experimental_memo
# def cache_load_ngram_lookup():
#     return load_ngram_lookup()


x_sum_dataset = cache_load_xsum_dataset()
# ngram_lookup = cache_load_ngram_lookup() # too large to cache


# @profile
def render_ngram_interface():
    # check if file exists
    build_flag = False
    if not exists(VOCABS_PATH):
        build_flag = True
    else:
        for n in range(1, MAX_N + 1):
            if not exists(NGRAM_PATH % n):
                build_flag = True
                break

    # build or load ngram
    if build_flag:
        ngram_lookup = build_ngram_lookup(x_sum_dataset)
    else:
        ngram_lookup = load_ngram_lookup()

    st.header("XSUM N-Gram Lookup")

    # input query
    query = st.text_input("Input query string (1 to 5 words):")

    # preprocess query
    pp_query_wrd = tuple(preprocess(query).split())
    st.write("Preprocessed query words:", pp_query_wrd)

    # ngram lookup
    lookup_dict = ngram_lookup.lookup(query_wrd=pp_query_wrd)
    case = lookup_dict["case"]
    matched_doc_idx = lookup_dict["match"]

    # write results
    st.write("**Search result:**")
    if case == 0:  # no query given
        st.write("No query given")
    elif case == 1:  # <unk> in query
        st.write("Unknown word in query")
    else:
        # st.write("case", case)
        # st.write(matched_doc_idx)
        results = [
            {
                "id": x_sum_dataset[int(doc_idx)]["id"],
                "document": x_sum_dataset[int(doc_idx)]["document"],
            }
            for doc_idx in matched_doc_idx
        ]
        st.write("* %d documents matched" % len(results))
        st.write(results)


if __name__ == "__main__":
    render_ngram_interface()
