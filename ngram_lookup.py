import string
import regex as rx

from collections import defaultdict  # set

from datasets import load_dataset
from tqdm import tqdm  # progress

from dictionary import Dictionary
from trie import Trie

TMP_LEN = 100
N = 3


def preprocess(text):  # TODO: replace it with tokenizer
    # strip
    out = text.strip()

    # lowercase
    out = out.lower()

    # remove punctuation
    out = out.translate(str.maketrans("", "", string.punctuation))  # email?, time?

    # remove control sequence
    out = rx.sub(r"\p{C}", " ", out)

    return out


class NgramLookup:
    def __init__(self, documents, ids):
        """
        Args:
            documents: A list, list of documents to build ngram upon
            ids:
        """
        self.documents = documents
        self.ids = ids

        self.dictionary = Dictionary()
        self.ngrams_root = {}

    def build_dictionary(self):
        print("Building dictionary...")
        for doc in tqdm(self.documents):
            words = doc.strip().split()
            for wrd in words:
                self.dictionary.add_wrd(wrd)

    def build_ngrams(self, n):
        """
        Generate ngrams from the given dictionary and store them in a dictionary
        {(ngram_idx):(idx)}

        Args:
            n: An integer, the rank of the grams that are generated

        Returns:
            A string, <unk> symbol if index does not exist else the word of the given index

        """

        # check if dictionary was built
        assert self.dictionary.get_num_of_words() != 1, "Build dictionary first"

        print("Generating %d-grams from documents" % n)

        ngram_to_idx_set = defaultdict(set)

        for id, doc in tqdm(zip(self.ids, self.documents)):
            words = doc.strip().split()
            for ngram in zip(words, words[1:], words[2:]):  # TODO: other than n=3
                ngram_idx = self.dictionary.get_idx_by_wrd_multiple(ngram)
                ngram_to_idx_set[ngram_idx].add(id)  # TODO: add idx instead of doc id

        self.ngrams_root[n] = ngram_to_idx_set

    def lookup(self, query_wrd):
        """

        Args:
            query_wrd: A tuple of query words

        Returns:
            A set of matched document ids, empty set if no match

        """
        query_idx = self.dictionary.get_idx_by_wrd_multiple(query_wrd)
        return self.ngrams_root[N][query_idx]

    # TODO: too slow, use in each document?
    def build_ngrams_trie(self, n):
        """
        Generate ngrams from the given dictionary and store them in a Trie data structure

        Args:
            n: An integer, the rank of the grams that are generated

        Returns:
            A string, <unk> symbol if index does not exist else the word of the given index

        """

        # check if dictionary was built
        assert self.dictionary.get_num_of_words() != 1, "Build dictionary first"

        unk_id = self.dictionary.get_unk_id()

        print("Generating %d-grams from documents" % n)

        trie = Trie()

        for doc in tqdm(self.documents):
            words = doc.strip().split()
            # new code
            for ngram in zip(words, words[1:], words[2:]):  # TODO: other than n=3
                ngram_idx = list(self.dictionary.get_idx_by_wrd_multiple(ngram))
                trie.add_ngram(ngram_idx)  # trigram_idx should be index

            # original code
            # ngram = [] # [start_id]
            # for wrd in words:
            #     if len(ngram) == n:
            #         trie.add_ngram(ngram)
            #         ngram = ngram[1:]
            #     idx = self.dictionary.get_idx_by_wrd(wrd)
            #     if idx == unk_id:
            #         self.unk_cnt += 1
            #     ngram.append(idx)
            # if len(ngram) == n:
            #     trie.add_ngram(ngram)
            #     ngram = ngram[1:]
            # trie.add_ngram(ngram)

        print("%d-grams are now stored in a Trie" % n)

        self.ngrams_root[n] = trie


if __name__ == "__main__":
    # Load dataset
    # TODO: use solution from https://github.com/cs6741/summary-analysis/issues/2 to load and align dataset
    dataset = load_dataset("xsum")

    # sample
    # document = dataset["train"][0]["document"]
    # summary = dataset["train"][0]["summary"]
    # id = dataset["train"][0]["id"]
    #
    # print(document)

    # preprocess sample document
    # pp_document = preprocess(document)

    # whole documents
    documents = dataset["train"]["document"][:TMP_LEN]
    ids = dataset["train"]["id"][:TMP_LEN]

    # preprocess whole documents
    pp_documents = list(map(preprocess, documents))

    # lookup
    ngram_lookup = NgramLookup(documents=pp_documents, ids=ids)

    # build dictionary
    ngram_lookup.build_dictionary()

    # query
    query = str(input("Enter query (3 words): "))  # TODO: now only 3
    query_wrd = tuple(preprocess(query).strip().split())

    # define N
    N = len(query_wrd)

    # build ngram
    ngram_lookup.build_ngrams(n=N)

    # lookup - returns ids of documents including query
    id_list = ngram_lookup.lookup(query_wrd=query_wrd)

    breakpoint()
