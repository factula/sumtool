from datasets import load_dataset

import csv

# TODO
# make a dataloader from the data_xsum: document input, summary output
# can we make a dataloader that merges all 3 info?

class XsumDatasetHandler():

    def __init__(self):
        self.__data_xsum = load_dataset("xsum") 
        self.__data_xsum_factuality = load_dataset("xsum_factuality","xsum_factuality")  # bbcid (int), system (string), summary (string), is_factual (class label), worker_id (string)
        self.__data_xsum_faithfulness = load_dataset("xsum_factuality","xsum_faithfulness")  # bbcid,system,summary,hallucination_type,hallucinated_span_start,hallucinated_span_end,worker_id
        self.train_dic = {}
        self._align_data()

    def _align_data(self):
        """schema till now:
        compiled_train_data: Dict(id, {
            'document': original document,
            'true_summary': true summary,
            'factuality_data': []
            'faithfulness_data':[]
        )
        """
        
        # TODO confirm the summary in xsum is the true summary

        for data in self.__data_xsum["train"]:
            self.train_dic[data["id"]] = {"document":data["document"], "true_summary":data["summary"], "factuality_data":[], "faithfulness_data":[]}

        for data in self.__data_xsum_factuality["train"]:
            if data["bbcid"] in self.train_dic:
                print("good")
                self.train_dic[data["bbcid"]]["factuality_data"].append("TODO: append detail here")

        for data in self.__data_xsum_faithfulness["train"]:
            if data["bbcid"] in self.train_dic:
                print("good")
                self.train_dic[data["bbcid"]]["faithfulness_data"].append("TODO: append detail here")

        # print(train_d['34687720'])


    


XsumDatasetHandler()

def compile_data():
    """Compiles the data from xsum dataset into a single dict.
    Returns:
        compiled_train_data: Dict(id, {
            'document': original document,
            'summary': generated summary,
            'factuality_data': [
                {
                    'worker_id': worker_id (str),
                    'is_factual': is_factual (bool),
                    'system': system (str),
                }
            ],
            'faithfulness_data': [
                {
                    'worker_id': worker_id (str),
                    'system': system (str),
                    'hallucination_type': hallucination_type (str),
                    'hallucinated_span': hallucinated_span (str),
                    'hallucinated_span_start': hallucinated_span_start (int),
                    'hallucinated_span_end': hallucinated_span_end (int),
                }
            ],
            'true_summary': true summary (str),
            }
        )
        xsum_train_data: Dict(id, {
            'document': original document,
            'summary': generated summary,
            }
        )
        xsum_val_data: Dict(id, {
            'document': original document,
            'summary': true summary,
            }
        )
        xsum_test_data: Dict(id, {
            'document': original document,
            'summary': true summary,
            }
        )
    """
    data_xsum = load_dataset("xsum")  # document (string), summary (string), id (string)
    data_xsum_factuality = load_dataset(
        "xsum_factuality"
    )  # bbcid (int), system (string), summary (string), is_factual (class label), worker_id (string)
    data_xsum_faithfulness = load_dataset(
        "xsum_factuality", "xsum_faithfulness"
    )  # bbcid,system,summary,hallucination_type,hallucinated_span_start,hallucinated_span_end,worker_id

    compiled_train_data = {}
    xsum_train_data = {}
    xsum_val_data = {}
    xsum_test_data = {}

    def handle_duplicates(category, new_version, id):
        if compiled_train_data[id][category] == new_version:
            return id
        print("Duplicate id: {}; {} does not match!".format(id, category))
        i = 0
        while True:
            if id + "_dup" + str(i) in compiled_train_data:
                if compiled_train_data[id + "_dup" + str(i)][category] == new_version:
                    print("Duplicate id: {}; adding to it...".format(id))
                    return id
                else:
                    print("Duplicate id: {}; {} does not match!".format(id, category))
                    i += 1
            else:
                id = id + "_dup" + str(i)
                print("New id: {}; adding it...".format(id))
                compiled_train_data[id] = {
                    "summary": summary,
                    "factuality_data": [],
                    "faithfulness_data": [],
                }
                return id

    for fact_d in data_xsum_factuality["train"]:
        id = str(fact_d["bbcid"])
        system = fact_d["system"]
        summary = fact_d["summary"]
        is_factual = fact_d["is_factual"]
        worker_id = fact_d["worker_id"]
        if id not in compiled_train_data:
            compiled_train_data[id] = {
                # 'summary': summary,
                "factuality_data": [],
                "faithfulness_data": [],
            }
        # id = handle_duplicates('summary', summary, id)
        compiled_train_data[id]["factuality_data"].append(
            {
                "worker_id": worker_id,
                "summary": summary,
                "is_factual": is_factual,
                "system": system,
            }
        )

    for faith_d in data_xsum_faithfulness["train"]:
        # id,system,summary,hallucination_type,hallucinated_span_start,hallucinated_span_end,worker_id
        id = str(faith_d["bbcid"])
        system = faith_d["system"]
        summary = faith_d["summary"]
        hallucination_type = faith_d["hallucination_type"]
        hallucinated_span_start = faith_d["hallucinated_span_start"]
        hallucinated_span_end = faith_d["hallucinated_span_end"]
        hallucinated_span = summary[hallucinated_span_start:hallucinated_span_end]
        worker_id = faith_d["worker_id"]
        if id not in compiled_train_data:
            print("(faithfulness) New id: {}; adding it...".format(id))
            compiled_train_data[id] = {
                # 'summary': summary,
                "factuality_data": [],
                "faithfulness_data": [],
            }
        # id = handle_duplicates('summary', summary, id)
        compiled_train_data[id]["faithfulness_data"].append(
            {
                "worker_id": worker_id,
                "system": system,
                "summary": summary,
                "hallucination_type": hallucination_type,
                "hallucinated_span": hallucinated_span,
                "hallucinated_span_start": hallucinated_span_start,
                "hallucinated_span_end": hallucinated_span_end,
            }
        )

    for xsum_d in data_xsum["train"]:
        id = xsum_d["id"]
        summary = xsum_d["summary"]
        document = xsum_d["document"]
        if id not in xsum_train_data:
            xsum_train_data[id] = {"summary": summary, "document": document}
            continue
        print("(xsum train) Duplicate id: {}; adding to it...".format(id))

    for xsum_d in data_xsum["validation"]:
        document = xsum_d["document"]
        summary = xsum_d["summary"]
        id = xsum_d["id"]
        if id not in xsum_val_data:
            xsum_val_data[id] = {"summary": summary, "document": document}
            continue
        print("(xsum val) Duplicate id: {}; skipping...".format(id))

    for xsum_d in data_xsum["test"]:
        document = xsum_d["document"]
        summary = xsum_d["summary"]
        id = xsum_d["id"]
        if id not in xsum_test_data:
            xsum_test_data[id] = {"summary": summary, "document": document}
            continue
        print("(xsum test) Duplicate id: {}; skipping...".format(id))

    return compiled_train_data, xsum_train_data, xsum_val_data, xsum_test_data


# compiled_train_data, xsum_train_data, xsum_val_data, xsum_test_data = compile_data()
# TODO
# issues I see:
# spans of hallucinations seem off by one character --> one-indexing vs zero-indexing?
# some summaries are shared across a document and diff worker ids. can/should we collapse them?  --> hypothesis: system and summary are correlated
# I think the structure of these dictionaries is far from optimal... think about it as we set up dataloader
