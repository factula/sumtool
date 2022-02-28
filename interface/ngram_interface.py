import streamlit as st

# from memory_profiler import profile
from os.path import exists, dirname, realpath, join

from sumtool.ngram import preprocess
from backend.viz_ngram_loader import (
    load_xsum_dataset,
    load_ngram_lookup,
    build_ngram_lookup,
    Case,
)

# parameters
CURRENT_PATH = dirname(realpath(__file__))  # current file path
VOCABS_PATH = join(CURRENT_PATH, "../sumtool/ngram/cache/vocabs")  # vocab file path
NGRAM_PATH = join(
    CURRENT_PATH, "../sumtool/ngram/cache/ngram_dict_%d"
)  # ngram file path
MIN_N = 1  # min n for ngram
MAX_N = 2  # max n for ngram
MAX_VOCAB_SIZE = 10000  # vocab size
SAVE_FLAG = True  # whether to save vocab, ngram files
NUM_PROC = 5  # # of processes to use for preprocessing


# @profile
def render_ngram_interface():
    # load dataset
    x_sum_dataset = load_xsum_dataset()

    # check if file exists
    build_flag = False
    if not exists(VOCABS_PATH):
        build_flag = True
    else:
        for n in range(MIN_N, MAX_N + 1):
            if not exists(NGRAM_PATH % n):
                build_flag = True
                break

    # build or load ngram
    if build_flag:
        ngram_lookup = build_ngram_lookup(
            x_sum_dataset, VOCABS_PATH, NGRAM_PATH, MAX_VOCAB_SIZE, MIN_N, MAX_N, SAVE_FLAG
        )
    else:
        ngram_lookup = load_ngram_lookup(VOCABS_PATH, NGRAM_PATH, MAX_VOCAB_SIZE, MIN_N, MAX_N)

    st.header("XSUM N-Gram Lookup")

    # input query
    query = st.text_input("Input query string (%d to %d words):" % (MIN_N, MAX_N))

    # preprocess query
    pp_query_wrd = tuple(preprocess(query).split())
    st.write("Preprocessed query words:", pp_query_wrd)

    # ngram lookup
    lookup_dict = ngram_lookup.lookup(query_wrd=pp_query_wrd)
    case = lookup_dict["case"]
    matched_doc_idx = lookup_dict["match"]

    # write results
    st.write("**Search result:**")
    if case == Case.no_query_given.value:
        st.write("No query given")
    elif case == Case.unk_in_query.value:
        st.write("Unknown word in query")
    else:
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
