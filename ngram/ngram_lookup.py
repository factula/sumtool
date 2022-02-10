import string
import regex as rx
import pickle
from os.path import exists

from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm  # progress bar

from dictionary import Dictionary


def preprocess(text):  # TODO: dictionary too big, replace it with tokenizer?
    # strip
    out = text.strip()

    # lowercase
    out = out.lower()

    # remove punctuation
    out = out.translate(str.maketrans("", "", string.punctuation))

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

        print("Generating %d-gram dictionary from documents" % n)

        ngram_to_idx_set = defaultdict(set)

        for doc_idx, doc in enumerate(tqdm(self.documents)):
            words = doc.strip().split()
            words = [self.dictionary.get_idx_by_wrd(word) for word in words]
            _ngrams = [words[k:] for k in range(n)]  # [words, words[1:], words[2:]]

            for ngram in zip(*_ngrams):
                ngram_to_idx_set[ngram].add(doc_idx)

        self.ngrams_root[n] = ngram_to_idx_set

    def lookup(self, query_wrd):
        """
        lookup given query from ngram dictionary

        Args:
            query_wrd: A tuple of query words

        Returns:
            A set of matched document indices, empty set if no match
        """
        query_idx = self.dictionary.get_idx_by_wrd_multiple(query_wrd)
        matched_doc_idx = self.ngrams_root[len(query_wrd)][
            query_idx
        ]  # document indices

        return matched_doc_idx

    def load_ngram_dict(self, n, file_path):
        """
        Load ngram dictionary from file

        Args:
            n: An integer, the rank of the grams
            file_path: A string, ngram dictionary file path
        """
        print("Loading %d-gram dictionary from '%s' ..." % (n, file_path))
        with open(file_path, "rb") as f:
            self.ngrams_root[n] = pickle.load(f)
        print("%d-gram dictionary length: %d" % (n, len(self.ngrams_root[n])))

    def save_ngram_dict(self, n, file_path):
        """
        Save ngram dictionary as a file

        Args:
            n: An integer, the rank of the grams
            file_path: A string, ngram dictionary file path
        """
        print("Saving %d-gram dictionary to '%s' ..." % (n, file_path))
        with open(file_path, "wb") as f:
            pickle.dump(self.ngrams_root[n], f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    DOC_LEN = 10000  # to test on smaller dataset
    VOCABS_PATH = "./cache/vocabs"
    NGRAM_FILE = "./cache/ngram_dict_%d.pkl"
    MAX_N = 5
    query = "Merry Christmas!"

    # Load dataset
    # TODO: use solution from https://github.com/cs6741/summary-analysis/issues/2 to load and align dataset
    dataset = load_dataset("xsum")

    # Parse dataset
    dataset = dataset["train"]
    documents = dataset["document"]
    ids = dataset["id"]

    # Preprocess documents
    pp_documents = list(map(preprocess, documents))

    # Build dictionary
    # if vocab file exists, load dictionary from file
    # else, build dictionary from corpus and save to VOCABS_PATH
    dictionary = Dictionary()
    if exists(VOCABS_PATH):
        dictionary.build_from_file(file_path=VOCABS_PATH)
    else:
        dictionary.build_from_corpus(corpus=pp_documents)
        dictionary.save_as_file(file_path=VOCABS_PATH)

    # Build ngram dictionary, n should be 1 - MAX_N
    # if ngram file exists, load ngram from file
    # else, build ngram from corpus and save to NGRAM_FILE
    ngram_lookup = NgramLookup(documents=pp_documents, ids=ids, dictionary=dictionary)
    for n in range(1, MAX_N + 1):
        if exists(NGRAM_FILE % n):
            ngram_lookup.load_ngram_dict(n=n, file_path=NGRAM_FILE % n)
        else:
            ngram_lookup.build_ngrams(n=n)
            ngram_lookup.save_ngram_dict(n=n, file_path=NGRAM_FILE % n)

    # Preprocess query
    print("query:", query)
    query_wrd = tuple(preprocess(query).strip().split())
    print("query_wrd:", query_wrd)

    # Lookup
    # returns indices of matched documents
    matched_idx_list = ngram_lookup.lookup(query_wrd=query_wrd)

    # Print results
    print("* %d matched results:" % len(matched_idx_list))
    for matched_idx in matched_idx_list:
        print("id:", ids[matched_idx])
        # print("document:", documents[matched_idx])

    # TODO: format input/output for visualization
