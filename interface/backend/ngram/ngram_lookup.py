import string
import regex as rx
import pickle
from os.path import exists

from collections import defaultdict
from tqdm import tqdm  # progress bar

from .dictionary import Dictionary

# from nltk.tokenize import word_tokenize, RegexpTokenizer
#
# def preprocess(text):
#     tokenizer = RegexpTokenizer(r"\w+")
#     pp_text = tokenizer.tokenize(text)
#     words = [word.lower() for word in pp_text]
#     return words


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
    out = out.strip().split()

    return out


class NgramLookup:
    def __init__(self, documents):
        """
        Args:
            documents: A list of documents to build ngram upon
        """
        self.documents = documents

        self.dictionary = Dictionary()
        self.ngrams_root = {}

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

    def build_ngram_dictionary(self, ngram_path, max_n, save_flag=True):
        """
        Build ngram dictionaries for n in (1, MAX_N + 1)
        if ngram file exists, load ngram from file
        else, build ngram from corpus and save to NGRAM_PATH

        Args:
            ngram_path: A string, path to ngram dictionaries file
            max_n: An integer, maximum rank of the grams
            save_flag: A boolean, whether to save as file
        """

        for n in range(1, max_n + 1):
            if exists(ngram_path % n):
                self.load_ngram_dict(n=n, file_path=ngram_path % n)
            else:
                self.build_ngrams(n=n)
                if save_flag:
                    self.save_ngram_dict(n=n, file_path=ngram_path % n)

        # check if dictionaries are built
        assert all(
            len(self.ngrams_root[n]) for n in range(1, max_n + 1)
        ), "ngram dictionaries were not built"

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
            indices = [self.dictionary.get_idx_by_wrd(word) for word in doc]
            _ngrams = [
                indices[k:] for k in range(n)
            ]  # if n=3, [indices, indices[1:], indices[2:]]

            for ngram in zip(*_ngrams):
                # do not add ngram if "<unk>" in ngram
                if self.dictionary.get_unk_idx() not in ngram:
                    ngram_to_idx_set[ngram].add(doc_idx)

        print("%d-gram dictionary length: %d" % (n, len(ngram_to_idx_set)))
        self.ngrams_root[n] = ngram_to_idx_set

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

    def lookup(self, query_wrd):
        """
        lookup given query from ngram dictionary

        Args:
            query_wrd: A tuple of query words

        Returns:
            A dictionary of {"case": int, "match": list}
            - case: which category given query belongs to
            - match: a list of matched document indices, empty if no match
        """
        print("Searching query...")

        # Case 0: return if no query given
        if len(query_wrd) == 0:
            return {"case": 0, "match": []}

        # transform words into indices
        query_idx = self.dictionary.get_idx_by_wrd_multiple(query_wrd)
        # print("query_idx:", query_idx)

        # Case 1: return if query includes <unk>
        if any(idx == self.dictionary.get_unk_idx() for idx in query_idx):
            return {"case": 1, "match": []}

        # lookup
        matched_doc_idx = self.ngrams_root[len(query_wrd)][
            query_idx
        ]  # document indices

        # Case 2: return matched document indices
        return {"case": 2, "match": matched_doc_idx}
