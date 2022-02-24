import json
import hashlib
import os
from typing import Dict, Optional, Any


def storage_dir(dataset):
    return f"./data/{dataset}"


def store_model_config(dataset: str, model: str, model_config: dict):
    model_config_json = json.dumps(model_config, sort_keys=True, indent=2)
    model_config_hash = hashlib.md5(model_config_json.encode()).hexdigest()

    dir = f"{storage_dir(dataset)}/models"
    path = f"{dir}/{model.replace('/', '-')}-{model_config_hash}.json"

    if not os.path.exists(path):
        os.makedirs(dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(model_config_json)

    return model_config_hash


def store_model_summaries(
    dataset: str,
    model: str,
    model_config: Dict,
    generated_summaries: Dict[str, str],
    metadata: Optional[Dict[str, Any]] = {},
):
    """
    Stores model summaries, indexed by dataset, model, model config hash & document id

    If the storage already has a summary for this document id and model config, it will overwrite it.
    Otherwise summaries are appended to the storage.

    Ex. /data/bert-base-summaries.json

    {
        "<model-config-hash>": {
            "<document-id>": {
                "summary": "this is awesome",
                "metadata": {
                    "tokens_ids": [1, 2, 3]",
                    "token_entropy": [0.5, 0.3, 0.2]
                }
            }
        }
    }

    Args:
        dataset: dataset name, i.e. "xsum"
        model: name of the model that was used to generate the summary (in huggingface model.config.name_or_path)
        model_config: dictionary of the config that was used to generate the summary (in huggingface serialize using model.config.to_dict())
        generated_summaries: dictionary of document id -> summary
        metadata: optional dict containing metadata for the generated summaries, for example token ids & entropy

    """

    model_config_hash = store_model_config(dataset, model, model_config)
    dir = storage_dir(dataset)
    path = f"{dir}/{model.replace('/', '-')}-summaries.json"

    if not os.path.exists(path):
        os.makedirs(dir, exist_ok=True)
        stored_summaries = {}
    else:
        with open(path, "r") as f:
            stored_summaries = json.load(f)

    for document_id, summary in generated_summaries.items():
        if model_config_hash not in stored_summaries:
            stored_summaries[model_config_hash] = {}
        stored_summaries[model_config_hash][str(document_id)] = {
            "summary": summary,
            "metadata": metadata[document_id] if document_id in metadata else {},
        }

    with open(path, "w") as f:
        f.write(json.dumps(stored_summaries, indent=2))


def store_summary_metrics(
    dataset: str,
    model: str,
    model_config_hash: str,
    summary_metrics: Dict[str, Dict[str, float]],
):
    """
    Stores summary metrics, indexed by dataset, model, model config hash & document id

    If the storage already has metrics for this document id and model config, it will overwrite it.
    Otherwise metrics are appended to the storage.

    Ex. /data/bert-base-metrics.json

    {
        "<model-config-hash>": {
            "<document-id>": {
                "bert-score": 0.5,
                "rouge-score": 0.5,
            }
        }
    }

    Args:
        dataset: dataset name, i.e. "xsum"
        model: name of the model that was used to generate the summary (in huggingface model.config.name_or_path)
        model_config_hash: hash of the model config that was used to generate the summary that is being evaluated
        summary_metrics: dictionary of document id -> {...summary metrics}

    """

    dir = storage_dir(dataset)
    path = f"{dir}/{model.replace('/', '-')}-metrics.json"

    if not os.path.exists(path):
        os.makedirs(dir, exist_ok=True)
        stored_summaries = {}
    else:
        with open(path, "r") as f:
            stored_summaries = json.load(f)

    for document_id, metrics in summary_metrics.items():
        if model_config_hash not in stored_summaries:
            stored_summaries[model_config_hash] = {}
        stored_summaries[model_config_hash][document_id] = metrics

    with open(path, "w") as f:
        f.write(json.dumps(stored_summaries, indent=2))


def store_summary_answers(
    # dataset: str,
    # model: str,
    # model_config_hash: str
):
    raise Exception("Not implemented!")
