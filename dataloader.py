from datasets import load_dataset
import csv

# TODO
# make a dataloader from the data_xsum: document input, summary output
# can we make a dataloader that merges all 3 info?

def compile_data():
    """ Compiles the data from xsum dataset into a single dict.
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
        val_data: Dict(id, {
            'document': original document,
            'summary': true summary,
            }
        )
        test_data: Dict(id, {
            'document': original document,
            'summary': true summary,
            }
        )
    """
    data_xsum = load_dataset("xsum") # document (string), summary (string), id (string)
    data_xsum_factuality = load_dataset("xsum_factuality") # bbcid (int), system (string), summary (string), is_factual (class label), worker_id (string)
    data_xsum_faithfulness = load_dataset("xsum_factuality", "xsum_faithfulness") # bbcid,system,summary,hallucination_type,hallucinated_span,hallucinated_span_start,hallucinated_span_end,worker_id

    compiled_train_data = {}
    val_data = {}
    test_data = {}

    def handle_duplicates(category, new_version, id):
        if (compiled_train_data[id][category] == new_version):
            return id
        print('Duplicate id: {}; {} does not match!'.format(id, category))
        i = 0
        while True:
            if (id+'_dup'+str(i) in compiled_train_data):
                if compiled_train_data[id+'_dup'+str(i)][category] == new_version:
                    print('Duplicate id: {}; adding to it...'.format(id))
                    return id
                else:
                    print('Duplicate id: {}; {} does not match!'.format(id, category))
                    i += 1
            else:
                id = id+'_dup'+str(i)
                print('New id: {}; adding it...'.format(id))
                compiled_train_data[id] = {
                    'summary': summary,
                    'factuality_data': [],
                    'faithfulness_data': []
                }
                return id

    for fact_d in data_xsum_factuality['train']:
        id = str(fact_d['bbcid'])
        system = fact_d['system']
        summary = fact_d['summary']
        is_factual = fact_d['is_factual']
        worker_id = fact_d['worker_id']
        if id not in compiled_train_data:
            compiled_train_data[id] = {
                            'summary': summary, 
                            'factuality_data': [],
                            'faithfulness_data': []
            }
        id = handle_duplicates('summary', summary, id)
        compiled_train_data[id]['factuality_data'].append(
        {
            'worker_id': worker_id,
            'is_factual': is_factual,
            'system': system
        })
            
    for faith_d in data_xsum_faithfulness['train']:
        # id,system,summary,hallucination_type,hallucinated_span_start,hallucinated_span_end,worker_id
        id = str(faith_d['bbcid'])
        system = faith_d['system']
        summary = faith_d['summary']
        hallucination_type = faith_d['hallucination_type']
        hallucinated_span_start = faith_d['hallucinated_span_start']
        hallucinated_span_end = faith_d['hallucinated_span_end']
        hallucinated_span = summary[hallucinated_span_start:hallucinated_span_end]
        worker_id = faith_d['worker_id']
        if id not in compiled_train_data:
            print('(faithfulness) New id: {}; adding it...'.format(id))
            compiled_train_data[id] = {
                                'summary': summary, 
                                'factuality_data': [],
                                'faithfulness_data': []
            }
        id = handle_duplicates('summary', summary, id)
        compiled_train_data[id]['faithfulness_data'].append(
        {
            'worker_id': worker_id,
            'system': system,
            'hallucination_type': hallucination_type,
            'hallucinated_span': hallucinated_span,
            'hallucinated_span_start': hallucinated_span_start,
            'hallucinated_span_end': hallucinated_span_end
        })

    for xsum_d in data_xsum['train']:
        id = xsum_d['id']
        summary = xsum_d['summary']
        document = xsum_d['document']
        if id not in compiled_train_data:
            print('(xsum train) New id: {}; adding it...'.format(id))
            compiled_train_data[id] = {
                'summary': None, 
                'factuality_data': [],
                'faithfulness_data': [],
            }
        compiled_train_data[id]['true_summary'] = summary
        compiled_train_data[id]['document'] = document

    for xsum_d in data_xsum['validation']:
        document = xsum_d['document']
        summary = xsum_d['summary']
        id = xsum_d['id']
        if id not in val_data:
            val_data[id] = {'summary': summary,
                            'document': document
            }
            continue
        print('(xsum val) Duplicate id: {}; skipping...'.format(id))

    for xsum_d in data_xsum['test']:
        document = xsum_d['document']
        summary = xsum_d['summary']
        id = xsum_d['id']
        if id not in test_data:
            test_data[id] = {'summary': summary,
                            'document': document
            }
            continue
        print('(xsum test) Duplicate id: {}; skipping...'.format(id))
            
    return compiled_train_data, val_data, test_data

compiled_train_data, val_data, test_data = compile_data()


# with open('data/compiled_data.csv', 'w') as csvfile:
#     fieldnames = ['id', 'split', 'system', 'worker_id', 'document', 'summary', 'is_factual', 'hallucination_type', 'hallucinated_span', 'hallucinated_span_start', 'hallucinated_span_end']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for id, data in compiled_data.items():
#         data_item = {'id': id, 'split': data['split'], 'system': data['system'], 'worker_id': data['worker_id'], 'document': data['document'], 'summary': data['summary'], 'is_factual': data['is_factual'], 'hallucination_type': data['hallucination_type'], 'hallucinated_span': data['hallucinated_span'], 'hallucinated_span_start': data['hallucinated_span_start'], 'hallucinated_span_end': data['hallucinated_span_end']}
#         writer.writerow(data_item)