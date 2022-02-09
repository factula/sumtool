import string
import regex as rx

from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm  # progress bar

from dictionary import Dictionary


def preprocess(text):  # TODO: replace it with tokenizer
    # strip
    out = text.strip()

    # lowercase
    out = out.lower()

    # remove punctuation
    out = out.translate(
        str.maketrans("", "", string.punctuation)
    )  # TODO: handle exceptions

    # remove control sequence
    out = rx.sub(r"\p{C}", " ", out)

    return out


class NgramLookup:
    def __init__(self, documents, ids, dictionary):
        """
        Args:
            documents: A list of documents to build ngram upon
            ids: A list of documents ids
            dictionary: Dictionary, pre-built dictionary object
        """
        self.documents = documents
        self.ids = ids

        self.dictionary = dictionary
        self.ngrams_root = {}

    def build_ngrams(self, n):
        """
        Generate ngrams from the given dictionary and store them in a dictionary
        - key: tuple of word indices
        - value: set of document indices

        Args:
            n: An integer, the rank of the grams that are generated
        """

        # check if dictionary was built
        assert self.dictionary.get_num_of_words() != 1, "Build dictionary first"

        # check if ngram was already built
        if n in self.ngrams_root:
            print("%d-grams are already built" % n)
            return

        print("Generating %d-grams from documents" % n)

        ngram_to_idx_set = defaultdict(set)

        for doc_idx, doc in enumerate(tqdm(self.documents)):
            words = doc.strip().split()
            _ngrams = [words[k:] for k in range(n)]
            for ngram in zip(*_ngrams):
                ngram_idx = self.dictionary.get_idx_by_wrd_multiple(ngram)
                ngram_to_idx_set[ngram_idx].add(doc_idx)

        self.ngrams_root[n] = ngram_to_idx_set

    def lookup(self, query_wrd):
        """

        Args:
            query_wrd: A tuple of query words

        Returns:
            A set of matched document indices, empty set if no match
        """
        query_idx = self.dictionary.get_idx_by_wrd_multiple(query_wrd)
        matched_doc_idx = self.ngrams_root[len(query_wrd)][
            query_idx
        ]  # document indices
        # matched_doc_id = [self.ids[doc_idx] for doc_idx in matched_doc_idx]  # map to document ids

        return matched_doc_idx


if __name__ == "__main__":
    TMP_LEN = 100000  # testing on small dataset
    VOCABS_FILE = None  # './vocab/vocabs.txt'

    # Load dataset
    # TODO: use solution from https://github.com/cs6741/summary-analysis/issues/2 to load and align dataset
    dataset = load_dataset("xsum")

    # Parse dataset
    dataset = dataset["train"]
    documents = dataset["document"]
    ids = dataset["id"]

    # preprocess documents
    pp_documents = list(map(preprocess, documents))

    # build dictionary
    dictionary = Dictionary()
    if VOCABS_FILE:
        dictionary.build_from_file(vocabs_file=VOCABS_FILE)
    else:
        dictionary.build_from_corpus(corpus=pp_documents)
        dictionary.save_as_file(file_path=VOCABS_FILE)

    # lookup
    ngram_lookup = NgramLookup(documents=pp_documents, ids=ids, dictionary=dictionary)

    while True:
        # input query
        query = str(input("* Enter query (press enter to stop): "))

        if not query:  # stop if empty
            break

        query_wrd = tuple(preprocess(query).strip().split())

        # build ngram
        ngram_lookup.build_ngrams(n=len(query_wrd))

        # lookup - returns indices of matched documents
        matched_idx_list = ngram_lookup.lookup(query_wrd=query_wrd)

        print("* %d matched results:" % len(matched_idx_list))
        for matched_idx in matched_idx_list:
            print("id:", ids[matched_idx])
            # print('document:', documents[matched_idx]) # too long

        # TODO: API for visualization
        # what should be the output - matched_id_list?
