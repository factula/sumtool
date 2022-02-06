"""
reference: https://github.com/mmz33/N-Gram-Language-Model/blob/master/trie.py
"""

from queue import Queue
from copy import copy


class Trie:
    """Trie data structure used to store ngrams and their frequencies
    Using trie, we can do queries such as finding ngrams, extracting counts, etc
    efficiently. Note also if you add trigrams for example, you are also able to extract information
    about smaller grams such as bigrams since prefix information is also stored

    Args:

    Returns:

    """

    def __init__(self):
        self.children = {}  # dict to store word indexes to represent ngrams
        self.freq = 0

    def add_ngram(self, ngram):
        """Add an ngram to the Trie

        Args:
          ngram: A list of word indexes representing an ngram

        Returns:

        """

        self.freq += 1
        if len(ngram) == 0:
            return
        idx = ngram[0]
        if idx not in self.children:
            self.children[idx] = Trie()
        next_node = self.children[idx]
        next_node.add_ngram(ngram[1:])

    def get_ngram_freq(self, ngram):
        """Retrieve the frequency of the given ngram

        Args:
          ngram: A list of word indexes representing an ngram

        Returns:
          An int, the frequency of the given ngram

        """

        if len(ngram) == 0:
            return self.freq
        idx = ngram[0]
        if idx not in self.children:
            return 0  # trigram not found so freq is 0
        next_node = self.children[idx]
        return next_node.get_ngram_freq(ngram[1:])

    def dfs(self, ngram, res, n, vocabulary):
        """Apply depth first search recursively to extract ngrams and their frequencies
        Note that this is slower than BFS and thus it is not mainly used
        (It is not well tested also)

        Args:
          ngram: A list representing an ngram
          res: A list to store the result of the search
          n: An integer, the rank of the gram
          vocabulary: An IndexMap, it can be either for corpus or vocabulary

        Returns:

        """

        # leaf node
        if len(ngram) == n:
            ngram_res = []
            for idx in ngram:
                ngram_res.append(vocabulary.get_wrd_by_idx(idx))
            res.append((ngram_res, self.get_freq()))

        for idx, child in self.get_children().items():
            ngram.append(idx)
            child.dfs(ngram, res, n, vocabulary)
            ngram = ngram[:-1]

    def bfs(self, n):
        """Apply a breadth first search to extract ngrams and their frequencies

        Args:
          n: An integer, the rank of the gram

        Returns:
          A list, the elements are pairs of ngrams and their frequencies

        """

        res = {}  # store ngrams with freq

        # queue for trie nodes
        q = Queue()
        q.put(self)  # add root

        # queue for ngrams
        ngrams_q = Queue()
        ngrams_q.put([])

        while not q.empty():
            u = q.get()  # curr node
            curr_ngram = ngrams_q.get()

            if len(curr_ngram) == n:
                ngram = tuple(curr_ngram)
                if ngram not in res:
                    res[ngram] = u.get_freq()
                else:
                    res[ngram] += u.get_freq()

            for idx, child in u.get_children().items():
                q.put(child)
                next_ngram = copy(curr_ngram)
                next_ngram.append(idx)
                ngrams_q.put(next_ngram)

        return res

    def get_children(self):
        """ """

        return self.children

    def get_num_of_children(self):
        """ """

        return len(self.children)

    def get_freq(self):
        """ """

        return self.freq

    def get_depth(self):
        """Compute the depth of the trie"""

        if len(self.children) == 0:
            return 0
        depth = 0
        for _, child in self.children.items():
            depth = max(depth, child.get_depth())
        return 1 + depth

    def get_ngram_last_node(self, ngram):
        """Return the last node in the trie for the given ngram
          It contains information about it's frequency

        Args:
          ngram: A list representing an ngram

        Returns:
          Trie Node if ngram exists and None otherwise

        """

        if len(ngram) == 0:
            return self
        idx = ngram[0]
        if idx not in self.children:
            return None  # ngram not found
        next_node = self.children[idx]
        return next_node.get_ngram_last_node(ngram[1:])
