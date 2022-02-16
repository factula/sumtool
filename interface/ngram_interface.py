import streamlit as st

from os.path import dirname, realpath, join
from datasets import load_dataset

from backend.ngram import NgramLookup, preprocess

# parameters
DOC_LEN = 200000  # to test on smaller dataset
CURRENT_PATH = dirname(realpath(__file__))  # current file path
VOCABS_PATH = join(CURRENT_PATH, "backend/ngram/cache/vocabs")  # vocab file path
NGRAM_PATH = join(CURRENT_PATH, "backend/ngram/cache/ngram_dict_%d.pkl")
MAX_N = 5
MAX_VOCAB_SIZE = 10000
SAVE_FLAG = True


def load_data_ngram_lookup():
    # TODO: replace it with sumtool
    x_sum_dataset = load_dataset("xsum")
    x_sum_dataset = x_sum_dataset["train"][:DOC_LEN]

    _documents = x_sum_dataset["document"]
    _ids = x_sum_dataset["id"]

    # Preprocess documents
    print("Preprocessing documents...")
    _pp_documents = list(map(preprocess, _documents))

    # ngram lookup
    ngram_lookup = NgramLookup(documents=_pp_documents)

    # build dictionary
    ngram_lookup.build_dictionary(
        vocabs_path=VOCABS_PATH, max_vocab_size=MAX_VOCAB_SIZE, save_flag=SAVE_FLAG
    )

    # build ngram dictionary
    ngram_lookup.build_ngram_dictionary(
        ngram_path=NGRAM_PATH, max_n=MAX_N, save_flag=SAVE_FLAG
    )

    # clear memory
    del ngram_lookup.documents

    return _documents, _ids, ngram_lookup


@st.experimental_memo
def cache_load_data_ngram_lookup():
    return load_data_ngram_lookup()


documents, ids, ngram_lookup = cache_load_data_ngram_lookup()


def render_ngram_interface():
    st.header("XSUM N-Gram Lookup")

    # input query
    query = st.text_input("Input query string (1 to 5 words):")

    # preprocess query
    pp_query_wrd = tuple(preprocess(query))
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
        results = [
            {"id": ids[doc_idx], "document": documents[doc_idx]}
            for doc_idx in matched_doc_idx
        ]
        st.write("* %d documents matched" % len(results))
        st.write(results)


if __name__ == "__main__":
    render_ngram_interface()
