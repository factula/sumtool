"""
reference: https://github.com/mmz33/N-Gram-Language-Model/blob/master/index_map.py
"""

from collections import defaultdict


class Dictionary:
    """
    Data structure for indexing words by unique ids
    It allows to retrieve queries in both direction (wrd->idx, and idx->wrd)
    """

    def __init__(self, vocabs_file=None):
        """
        Args:
            vocabs_file: A string, vocabs file path to load
        """
        self.idx = 0  # current index for each word

        self.wrd_to_idx = {}
        self.idx_to_wrd = {}
        self.wrd_freq = defaultdict(int)

        if vocabs_file:
            with open(vocabs_file, "r") as vocabs:
                for wrd in vocabs:
                    self.add_wrd(wrd.strip())
        else:
            # <unk> already exist in the vocabulary
            for wrd in {"<unk>"}:
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

    def get_idx_by_wrd(self, wrd):
        """
        Return the index of the given word

        Args:
          wrd: A string, the word of the given index

        Returns:
          An integer, -1 if word does not exist else the index of the word

        """

        if wrd not in self.wrd_to_idx:
            return self.get_unk_id()
        return self.wrd_to_idx[wrd]

    def get_num_of_words(self):
        """
        Return the total number of words

        Returns:
          :return: An integer, the number of total words

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
