import string
import math
import regex as rx
from os.path import exists

from tqdm import tqdm  # progress bar

from os.path import dirname, realpath, join
from datasets import load_dataset

import threading

import pyarrow as pa
import pyarrow.parquet as pq
from microdict import mdict

from .dictionary import Dictionary
from enum import Enum


class LookupCase(Enum):
    no_query_given = 0
    unk_in_query = 1
    match_not_found = 2
    match_found = 3


def preprocess(text):
    # strip
    out = text.strip()

    # lowercase
    out = out.lower()

    # remove punctuation
    out = out.translate(str.maketrans("", "", string.punctuation))

    # remove control sequence
    out = rx.sub(r"\p{C}", " ", out)

    # split into list
    # out = out.strip().split()

    return out


class NgramLookup:
    def __init__(self, documents):
        """
        Args:
            documents: A list of documents to build ngram upon
        """
        self.documents = documents

        self.dictionary = Dictionary()
        self.unk_idx = 0
        self.ngrams_root = {}

        # key to build ngram int id
        self.MAX = self._generate_int_key(self.tokenizer.vocab_size)

    def _generate_int_key(self, x):
        # check if x is power of 10
        if (10 ** math.log(x, 10)) == x:
            return x
        else:
            digit = len(str(x))
            x -= x % -(10 ** digit)
            return x

    def build_dictionary(self, vocabs_path, max_vocab_size, save_flag=True):
        """
        Build dictionary
        if vocabs file exists, load dictionary from file
        else, build dictionary from corpus and save to VOCABS_PATH

        Args:
            vocabs_path: A string, path to vocabs file
            max_vocab_size: An integer, maximum size of vocabulary
            save_flag: A boolean, whether to save as file
        """

        if exists(vocabs_path):
            self.dictionary.build_from_file(file_path=vocabs_path)
        else:
            self.dictionary.build_from_corpus(corpus=self.documents)
            self.dictionary.limit_max_vocab_size(max_vocab_size=max_vocab_size)
            if save_flag:
                self.dictionary.save_as_file(file_path=vocabs_path)

    def build_ngram_dictionary(self, ngram_path, min_n, max_n, save_flag=True):
        """
        Build ngram dictionaries for n in (min_n, max_n + 1)
        if ngram file exists, load ngram from file
        else, build ngram from corpus and save to NGRAM_PATH

        Args:
            ngram_path: A string, path to ngram dictionaries file
            min_n: An integer, minimum rank of the grams
            max_n: An integer, maximum rank of the grams
            save_flag: A boolean, whether to save as file
        """

        for n in range(min_n, max_n + 1):
            if exists(ngram_path % n):
                self.load_ngram_dict(n=n, file_path=ngram_path % n)
            else:
                self.build_ngrams(n=n)
                if save_flag:
                    self.save_ngram_dict(n=n, file_path=ngram_path % n)

        # check if dictionaries are built
        assert all(
            len(self.ngrams_root[n]) for n in range(min_n, max_n + 1)
        ), "ngram dictionaries are not built"

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

        # maps from ngram_int_id to idx
        ngram_to_idx_set = mdict.create("i64:i32")
        # list of document ids for idx
        doc_table = []

        for doc_idx, doc in enumerate(tqdm(self.documents)):
            indices = [self.dictionary.get_idx_by_wrd(word) for word in doc.split()]

            start = 0
            for word_idx in indices:
                if word_idx == self.unk_idx:  # if word is <unk>, pass
                    start = 0
                    continue

                start = start % (self.MAX ** (n - 1))
                start = self.MAX * start + word_idx

                if start < (self.MAX ** (n - 1)):
                    # print("too early, continue")
                    continue

                # use doc_table as [set], and ngram_to_idx_set as {ngram: idx}
                if start not in ngram_to_idx_set:
                    ngram_to_idx_set[start] = len(doc_table)
                    doc_table.append(set([doc_idx]))
                else:  # ngram already exists
                    doc_table[ngram_to_idx_set[start]].add(doc_idx)

        # transform into pyarrow.Table
        ngram = pa.array([k for k in ngram_to_idx_set], pa.int64())
        doc_idx_list = pa.array(
            [list(doc_table[v]) for v in ngram_to_idx_set.values()],
            pa.list_(pa.int32()),
        )
        # Important: need to align ngram_to_idx_set value and doc_table!

        table = pa.Table.from_arrays(
            [ngram, doc_idx_list],
            schema=pa.schema(
                [
                    pa.field("ngram", ngram.type),
                    pa.field("doc_idx_list", doc_idx_list.type),
                ]
            ),
        )

        print("%d-gram dictionary length: %d" % (n, len(ngram_to_idx_set)))
        self.ngrams_root[n] = table

    def load_ngram_dict(self, n, file_path):
        """
        Load ngram dictionary from file

        Args:
            n: An integer, the rank of the grams
            file_path: A string, ngram dictionary file path
        """

        print("Loading %d-gram dictionary from '%s' ..." % (n, file_path))

        # save as parquet
        self.ngrams_root[n] = pq.read_table(source=file_path)
        # print(self.ngrams_root[n])
        print("%d-gram dictionary length: %d" % (n, len(self.ngrams_root[n])))

    def save_ngram_dict(self, n, file_path):
        """
        Save ngram dictionary as a file

        Args:
            n: An integer, the rank of the grams
            file_path: A string, ngram dictionary file path
        """

        print("Saving %d-gram dictionary to '%s' ..." % (n, file_path))

        # parquet
        pq.write_table(self.ngrams_root[n], file_path)

    def lookup(self, query_wrd):
        """
        lookup given query from ngram dictionary

        Args:
            query_wrd: A list of query words

        Returns:
            A dictionary of {"case": int, "match": list}
            - case: which category given query belongs to
            - match: a list of matched document indices, empty if no match
        """
        print("Searching query...")
        n = len(query_wrd)

        # Case 0: return if no query given
        if n == 0:
            return {"case": 0, "match": []}

        # transform words into indices
        query_idx = self.dictionary.get_idx_by_wrd_multiple(query_wrd)
        print("query_idx:", query_idx)

        # Case 1: return if query includes <unk>
        if any(idx == self.dictionary.get_unk_idx() for idx in query_idx):
            return {"case": 1, "match": []}

        ngram_int = 0
        # ngram_int = query_idx[0] * (MAX ** 2) + query_idx[1] * (MAX ** 1) + query_idx[2] * (MAX ** 0)
        for i, idx in enumerate(query_idx):
            ngram_int += idx * (self.MAX ** (n - 1 - i))

        df = self.ngrams_root[n].to_pandas()

        matched_doc_idx = df.loc[df["ngram"] == ngram_int, "doc_idx_list"]

        if len(matched_doc_idx) == 0:
            # Case 2: all words are in vocabs but no match found
            return {"case": 2, "match": []}
        else:
            # Case 3: match found- return matched document indices
            return {"case": 3, "match": list(matched_doc_idx.item())}


# @profile
def main():
    CURRENT_PATH = dirname(realpath(__file__))  # current file path
    VOCABS_PATH = join(CURRENT_PATH, "cache/vocabs")
    NGRAM_PATH = join(CURRENT_PATH, "cache/ngram_dict_%d")
    MIN_N = 1
    MAX_N = 4
    MAX_VOCAB_SIZE = 10000
    SAVE_FLAG = True

    # TODO: replace it with sumtool
    x_sum_dataset = load_dataset("xsum")
    x_sum_dataset = x_sum_dataset["train"]

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
        target=ngram_lookup.build_ngram_dictionary,
        args=(NGRAM_PATH, MIN_N, MAX_N, SAVE_FLAG),
    )
    p2.start()
    p2.join()

    # this has to be moved to "interface/ngram_interface.py"
    # query = "to the"
    #
    # # preprocess query
    # pp_query_wrd = tuple(preprocess(query).split())
    # print(pp_query_wrd)
    #
    # # ngram lookup
    # lookup_dict = ngram_lookup.lookup(query_wrd=pp_query_wrd)
    #
    # print(lookup_dict["case"])
    # print(len(lookup_dict["match"]))


if __name__ == "__main__":
    main()
