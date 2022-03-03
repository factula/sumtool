from collections import defaultdict
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
            tokenizer: TODO
        """
        self.documents = documents

        self.tokenizer = tokenizer
        self.unk_idx = tokenizer.unk_token_id
        self.ngrams_root = {}

        # key to build ngram int id
        self.MAX = round(self.tokenizer.vocab_size, -5)  # bart tokenizer size is 50265

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

    def build_ngrams(self, n):  # now just bigram
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

        for doc_idx, doc in enumerate(tqdm(self.documents)):
            indices = self.tokenizer.encode(doc, add_special_tokens=False)

            # print(indices)
            # print(type(indices))

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

    def lookup(self, query_idx, df):
        """
        lookup given query from ngram dictionary (pandas dataframe)

        Args:
            query_idx: A list of query indices
            df: pa.Dataframe, TODO

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

        ngram_int = 0
        # ngram_int = query_idx[0] * (MAX ** 2) + query_idx[1] * (MAX ** 1) + query_idx[2] * (MAX ** 0)
        for i, idx in enumerate(query_idx):
            ngram_int += idx * (self.MAX ** (n - 1 - i))

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
        print("Looking up summary ngrams from the training set")
        summary_indices = self.tokenizer.encode(summary, add_special_tokens=False)

        df = self.ngrams_root[n].to_pandas()

        result_dict_list = []
        for summary_ngram in zip(summary_indices, summary_indices[1:]):  # only bigram
            result_dict = self.lookup(summary_ngram, df)
            result_dict_list.append(
                {
                    "ngram": summary_ngram,
                    "case": result_dict["case"],
                    "match": result_dict["match"],
                }
            )
        return result_dict_list

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
        print("Looking up summary ngrams from the given document")
        summary_indices = self.tokenizer.encode(summary, add_special_tokens=False)
        document_indices = self.tokenizer.encode(document, add_special_tokens=False)

        # TODO: add word idx to document_ngram_dict?, now add count
        document_ngram_dict = defaultdict(set)
        for document_ngram in zip(
            document_indices, document_indices[1:]
        ):  # only bigram
            if document_ngram in document_ngram_dict:
                document_ngram_dict[document_ngram] += 1
            else:
                document_ngram_dict[document_ngram] = 1

        # print(document_ngram_dict)

        result_dict_list = []  # return count
        for summary_ngram in zip(summary_indices, summary_indices[1:]):  # only bigram
            # Case 1: return if query includes <unk>
            if any(idx == self.unk_idx for idx in summary_ngram):
                result_dict_list.append(
                    {"ngram": summary_ngram, "case": 1, "match": []}
                )
                continue

            # Case 2: all words are in vocabs but no match found
            if summary_ngram not in document_ngram_dict:
                result_dict_list.append(
                    {"ngram": summary_ngram, "case": 2, "match": []}
                )
            else:
                # Case 3: match found- return matched document indices
                result_dict_list.append(
                    {
                        "ngram": summary_ngram,
                        "case": 3,
                        "match": document_ngram_dict[summary_ngram],
                    }
                )
        return result_dict_list


# @profile
def main():
    CURRENT_PATH = dirname(realpath(__file__))  # current file path
    NGRAM_PATH = join(CURRENT_PATH, "cache_bart_tokenizer/ngram_dict_%d")
    MIN_N = 2
    MAX_N = 2  # just bigram
    SAVE_FLAG = True

    # TODO: replace it with sumtool
    x_sum_dataset = load_dataset("xsum")
    # test_dataset = x_sum_dataset["test"]
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
    ngram_lookup = NgramSummaryLookup(
        documents=x_sum_dataset["document"], tokenizer=tokenizer
    )

    # build ngram dictionary with multithreading
    p2 = threading.Thread(
        target=ngram_lookup.build_ngram_dictionary,
        args=(NGRAM_PATH, MIN_N, MAX_N, SAVE_FLAG),
    )
    p2.start()
    p2.join()

    # this has to be moved to interface
    document = (
        "Prison Link Cymru had 1,099 referrals in 2015-16 and said some ex-offenders were living rough for up "
        "to a year before finding suitable accommodation.\nWorkers at the charity claim investment in housing "
        "would be cheaper than jailing homeless repeat offenders.\nThe Welsh Government said more people than "
        "ever were getting help to address housing problems.\nChanges to the Housing Act in Wales, introduced "
        "in 2015, removed the right for prison leavers to be given priority for accommodation.\nPrison Link "
        "Cymru, which helps people find accommodation after their release, said things were generally good for "
        "women because issues such as children or domestic violence were now considered.\nHowever, "
        "the same could not be said for men, the charity said, because issues which often affect them, "
        "such as post traumatic stress disorder or drug dependency, were often viewed as less of a "
        "priority.\nAndrew Stevens, who works in Welsh prisons trying to secure housing for prison leavers, "
        'said the need for accommodation was "chronic".\n"There\'s a desperate need for it, finding suitable '
        'accommodation for those leaving prison there is just a lack of it everywhere," he said.\n"It could '
        'take six months to a year, without a lot of help they could be on the streets for six months.\n"When '
        "you think of the consequences of either being on the street, especially with the cold weather at the "
        'moment or you may have a roof over your head, sometimes there is only one choice."\nMr Stevens '
        'believes building more one-bedroom flats could help ease the problem.\n"The average price is a hundred '
        "pounds a week to keep someone in a rented flat, prison is a lot more than that so I would imagine it "
        'would save the public purse quite a few pounds," he said.\nOfficial figures show 830 one-bedroom '
        "properties were built in the year to March 2016, of an overall total of 6,900 new properties in "
        "Wales.\nMarc, 50, who has been in and out of prison for the past 20 years for burglary offences, "
        "said he struggled to find accommodation each time he was released.\nHe said he would ask himself: "
        '"Where am I going to stay? Where am I going to live? Have I got somewhere where I can see my '
        'daughter."\n"You\'re put out among the same sort of people doing the same sort of thing, '
        "and it's difficult, it's difficult to get away from it. It's like every man for himself, "
        "there's nothing.\"\nMarc has now found stable accommodation with homeless charity Emmaus and said it "
        "had been life changing.\n\"You feel safe, you got hot food, you've got company of people in similar "
        "situations to yourself but all dealing with different issues. It's a constructive, "
        'helpful atmosphere," he said.\nTom Clarke, chief executive of Emmaus South Wales, agreed there was not '
        "enough support available.\n\"We do still see [people] homeless on the streets, so clearly they haven't "
        'got accommodation and haven\'t got provision," he said.\n"I think the key is connecting people with '
        "the services they need. I don't delude myself that Emmaus can offer a one size fits all for everyone, "
        "we can't.\n\"But there must be other opportunities and given suitable encouragement I believe that can "
        'and should happen."\nA Welsh Government spokesman said the national pathway for homeless services to '
        "children, young people and adults in the secure estate had prevented many people from losing their "
        "home whilst serving their prison sentence.\nIt added there were already significant demands for "
        "one-bedroom flats across the public and private sector and it was providing 20,000 new affordable "
        "homes in the next five years."
    )
    summary = 'There is a "chronic" need for more housing for prison leavers in Wales, according to a charity.'

    summary_from_dataset = ngram_lookup.lookup_summary_from_dataset(summary, n=2)
    summary_from_document = ngram_lookup.lookup_summary_from_document(
        summary, document, n=2
    )  # only bigram

    # print(summary_from_dataset) # too long to print
    # for l in summary_from_dataset:
    #     print(l["ngram"], l["case"], len(l["match"]))
    # print(summary_from_document)


if __name__ == "__main__":
    main()
