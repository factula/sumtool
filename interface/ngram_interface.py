from genericpath import exists
import streamlit as st

from memory_profiler import profile
from os.path import dirname, realpath, join
from datasets import load_dataset

from backend.ngram import NgramLookup, preprocess

# import backend.ngram.ngram_lookup as nl

# parameters
CURRENT_PATH = dirname(realpath(__file__))  # current file path
VOCABS_PATH = join(CURRENT_PATH, "backend/ngram/cache/vocabs")  # vocab file path
NGRAM_PATH = join(CURRENT_PATH, "backend/ngram/cache_singlethread/ngram_dict_%d")
MAX_N = 4
MAX_VOCAB_SIZE = 10000
# SAVE_FLAG = True

# nl.main()

# file_path = str(join(CURRENT_PATH, "backend/ngram/ngram_lookup.py"))
# with open(file_path) as f:
#     code = compile(f.read(), file_path, 'exec')
#     exec(code)

# file_path = join(CURRENT_PATH, "backend/ngram/ngram_lookup.py")
# print(file_path)
# exec(open(file_path).read())


def load_xsum_dataset():
    x_sum_dataset = load_dataset("xsum")
    x_sum_dataset = x_sum_dataset["train"]

    return x_sum_dataset


def load_data_ngram_lookup():
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
# def cache_load_data_ngram_lookup():
#     return load_data_ngram_lookup()


x_sum_dataset = cache_load_xsum_dataset()
# ngram_lookup = cache_load_data_ngram_lookup()


@profile
def render_ngram_interface():
    # x_sum_dataset = load_xsum_dataset()
    ngram_lookup = load_data_ngram_lookup()

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
