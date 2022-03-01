from os.path import exists

from tqdm import tqdm  # progress bar

from os.path import dirname, realpath, join
from datasets import load_dataset

import threading

import pyarrow as pa
import pyarrow.parquet as pq
from microdict import mdict

# newly added for ngram-summary-lookup
from transformers import BartTokenizer


def load_tokenizer():
    return BartTokenizer.from_pretrained("facebook/bart-large-xsum")


class NgramSummaryLookup:
    def __init__(self, documents, tokenizer):
        """
        Args:
            documents: A list of documents to build ngram upon
        """
        self.documents = documents

        self.tokenizer = tokenizer
        self.unk_idx = tokenizer.unk_token_id
        self.ngrams_root = {}

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

    def build_ngram(self, n):  # now just bigram
        """
        Generate ngrams from the given dictionary and store them in a pyarrow.Table

        Args:
            tokenizer: A PreTrainedTokenizer(), now using 'facebook/bart-large-xsum'
            n: An integer, the rank of the grams that are generated
        """

        # check if ngram was already built
        if n in self.ngrams_root:
            print("%d-grams are already built" % n)
            return

        print("Generating %d-gram dictionary from documents" % n)

        # maps from ngram_int_id to idx
        ngram_to_idx_set = mdict.create("i64:i32")
        # list of document ids for idx
        doc_table = []

        MAX = round(self.tokenizer.vocab_size, -5)  # bart tokenizer size - 50265

        for doc_idx, doc in enumerate(tqdm(self.documents)):
            indices = self.tokenizer.encode(doc, add_special_tokens=False)

            # print(indices)
            # print(type(indices))

            start = 0
            for word_idx in indices:
                if word_idx == self.unk_idx:  # if word is <unk>, pass
                    start = 0
                    continue

                start = start % (MAX ** (n - 1))
                start = MAX * start + word_idx

                if start < (MAX ** (n - 1)):
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

        # load parquet file
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

        # save as parquet file
        pq.write_table(self.ngrams_root[n], file_path)

    def lookup(self, query_idx, df):  # TODO
        """
        lookup given query from ngram dictionary

        Args:
            query_idx: A list of query indices
            df:

        Returns: TODO
            A dictionary of {"case": int, "match": list}
            - case: which category given query belongs to
            - match: a list of matched document indices, empty if no match
        """

        n = len(query_idx)

        # Case 0: return if no query given
        if n == 0:
            return {"case": 0, "match": []}

        # Case 1: return if query includes <unk>
        if any(idx == self.unk_idx for idx in query_idx):
            return {"case": 1, "match": []}

        MAX = round(
            self.tokenizer.vocab_size, -5
        )  # bart tokenizer size - 50265 # TODO: change it to self.MAX?

        ngram_int = 0
        # ngram_int = query_idx[0] * (MAX ** 2) + query_idx[1] * (MAX ** 1) + query_idx[2] * (MAX ** 0)
        for i, idx in enumerate(query_idx):
            ngram_int += idx * (MAX ** (n - 1 - i))

        matched_doc_idx = df.loc[df["ngram"] == ngram_int, "doc_idx_list"]

        if len(matched_doc_idx) == 0:
            # Case 2: all words are in vocabs but no match found
            return {"case": 2, "match": []}
        else:
            # Case 3: match found- return matched document indices
            return {"case": 3, "match": list(matched_doc_idx.item())}

    def lookup_summary_from_dataset(self, summary, n):
        """
        lookup given summary from the whole training dataset

        Args:
            summary: A String, generated summary
            n:

        Returns: TODO
            A dictionary of {"case": int, "match": list}
            - case: which category given query belongs to
            - match: a list of matched document indices, empty if no match
        """
        print("Looking up summary from the training set")
        summary_indices = self.tokenizer.encode(summary, add_special_tokens=False)

        df = self.ngrams_root[n].to_pandas()  # TODO: change it to self.df?

        for summary_ngram in zip(summary_indices, summary_indices[1:]):  # only bigram
            self.lookup(summary_ngram, df)

    def lookup_summary_from_document(self, summary, document, n):
        """
        lookup given summary from the given document

        Args:
            summary: A String, summary generated from the document
            document: A String

        Returns: TODO
            A dictionary of {"case": int, "match": list}
            - case: which category given query belongs to
            - match: a list of matched document indices, empty if no match
        """
        print("Looking up summary from the training set")
        summary_indices = self.tokenizer.encode(summary, add_special_tokens=False)
        document_indices = self.tokenizer.encode(document, add_special_tokens=False)

        df = self.ngrams_root[n].to_pandas()  # TODO: change it to self.df?

        # TODO: build document ngram dictionary
        document_ngram_dictionary = dict()

        for summary_ngram in zip(summary_indices, summary_indices[1:]):  # only bigram
            # lookup_from_dictionary(summary_ngram, df)
            pass

        return


# @profile
def main():
    CURRENT_PATH = dirname(realpath(__file__))  # current file path
    NGRAM_PATH = join(CURRENT_PATH, "cache_bart_tokenizer/ngram_dict_%d")
    MIN_N = 2
    MAX_N = 2  # just bigram
    SAVE_FLAG = True

    # TODO: replace it with sumtool
    x_sum_dataset = load_dataset("xsum")
    x_sum_dataset = x_sum_dataset["train"]

    # TODO: put tokenizer into NgramLookup?
    tokenizer = load_tokenizer()

    # TODO: how to tokenize?, now 2
    # 1. store encoded document to x_sum_dataset and pass it to ngram_lookup.documents
    # 2. pass the whole document to ngram_lookup and encode with internal for loop

    # for train_data in x_sum_dataset:
    #     doc = train_data["document"]
    #     wrd_idx_list = tokenizer.encode(doc, add_special_tokens=False)

    # Preprocess documents
    # print("Tokenizing documents...")
    # x_sum_dataset = x_sum_dataset.map(
    #     lambda example: {"pp_document": preprocess(example["document"])}, num_proc=5
    # )

    # ngram lookup
    ngram_lookup = NgramSummaryLookup(documents=x_sum_dataset["document"])

    # build ngram dictionary with multithreading
    p2 = threading.Thread(
        target=ngram_lookup.build_ngram_dictionary,
        args=(NGRAM_PATH, MIN_N, MAX_N, SAVE_FLAG),
    )
    p2.start()
    p2.join()

    # this has to be moved to interface
    summary = None  # TODO
    document = None  # TODO

    ngram_lookup.lookup_summary_from_dataset(summary, n=2)  # only bigram
    ngram_lookup.lookup_summary_from_document(summary, document, n=2)  # only bigram


if __name__ == "__main__":
    main()
