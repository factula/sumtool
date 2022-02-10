"""
reference: https://github.com/mmz33/N-Gram-Language-Model/blob/master/index_map.py
"""

from collections import defaultdict
from tqdm import tqdm  # progress


class Dictionary:
    """
    Data structure for indexing words by unique ids
    It allows to retrieve queries in both direction (wrd->idx, and idx->wrd)
    """

    def __init__(self):
        self.idx = 0  # current index for each word

        self.wrd_to_idx = {}
        self.idx_to_wrd = {}
        self.wrd_freq = defaultdict(int)

    def build_from_file(self, file_path):
        """
        Args:
            file_path: A string, vocabs file path to load
                         lines of {word} \t {freq}
                         sorted in descending order with freq
        """
        print("Loading dictionary from '%s' ..." % file_path)
        with open(file_path, "r") as f:
            lines = f.read().splitlines()
            for idx, line in enumerate(lines):
                wrd = line.split()[0]
                freq = int(line.split()[1])

                self.add_wrd(wrd)  # add word
                self.wrd_freq[idx] = freq  # add freq

        # set total number of words
        self.idx = len(self.wrd_to_idx)

    def build_from_corpus(self, corpus):
        """
        build dictionary from corpus

        Args:
            corpus: A list, list of documents/sentences to build a dictionary upon
        """
        print("Building dictionary from corpus...")
        self.add_wrd(self.get_unk_wrd())  # add <unk> token
        for doc in tqdm(corpus):
            words = doc.strip().split()
            for wrd in words:
                self.add_wrd(wrd)

    @staticmethod
    def get_unk_wrd():
        return "<unk>"

    def get_unk_id(self):
        return self.wrd_to_idx[self.get_unk_wrd()]

    def add_wrd(self, wrd):
        """
        Update dictionary and increase word's frequency

        Args:
            wrd: A string, the input word

        """

        if wrd not in self.wrd_to_idx:
            self.wrd_to_idx[wrd] = self.idx
            self.idx_to_wrd[self.idx] = wrd
            self.idx += 1

        self.wrd_freq[self.wrd_to_idx[wrd]] += 1

    def get_wrd_by_idx(self, idx):
        """
        Return the word of the given index

        Args:
            idx: An int, the index of the given word

        Returns:
            A string, <unk> symbol if index does not exist else the word of the given index

        """

        if idx not in self.idx_to_wrd:
            return self.get_unk_wrd()
        return self.idx_to_wrd[idx]

    def get_wrd_by_idx_multiple(self, idx_tuple):
        """
        Return the word tuple of the given indices

        Args:
            idx_tuple: A tuple of indices

        Returns:
            A tuple of corresponding words, <unk> symbol if index does not exist else the word of the given index

        """

        return tuple(map(self.get_wrd_by_idx, idx_tuple))

    def get_idx_by_wrd(self, wrd):
        """
        Return the index of the given word

        Args:
            wrd: A string, the word

        Returns:
            An integer, get_unk_id() if word does not exist else the index of the word

        """

        if wrd not in self.wrd_to_idx:
            return self.get_unk_id()
        return self.wrd_to_idx[wrd]

    def get_idx_by_wrd_multiple(self, wrd_tuple):
        """
        Return the index tuple of the given word tuple

        Args:
            wrd_tuple: A tuple of words

        Returns:
            A tuple of corresponding indices

        """

        return tuple(map(self.get_idx_by_wrd, wrd_tuple))

    def get_num_of_words(self):
        """
        Return the total number of words

        Returns:
            An integer, the number of total words

        """

        return self.idx

    def get_wrd_freq_by_idx(self, idx):
        """
        Return the frequency of a word given it's index

        Args:
            idx: An integer, an index of a word

        Returns:
            An integer, the frequency of the given word

        """

        return self.wrd_freq[idx]

    def get_wrd_freq(self, wrd):
        """
        Return the frequency of a word

        Args:
            wrd: A string

        Returns:
            An integer, the frequency of the given word

        """

        return self.wrd_freq[self.get_idx_by_wrd(wrd)]

    def get_wrd_freq_items(self):

        return self.wrd_freq.items()

    def save_as_file(self, file_path):
        sorted_wrd_freq = self.sort_dict_by_value(self.wrd_freq)

        print("Saving dictionary to %s" % file_path)
        with open(file_path, "w") as f:
            for idx, freq in sorted_wrd_freq.items():
                f.write("%s\t%d\n" % (self.get_wrd_by_idx(idx), freq))

