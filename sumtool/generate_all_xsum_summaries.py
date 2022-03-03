import argparse
import datasets
from sumtool.generate_xsum_summary import (
    load_summarization_model_and_tokenizer,
    generate_summaries,
)
from sumtool.xsum_dataset import XsumDataset
from sumtool.storage import store_model_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run inference on an xsum example using a pre-trained model"
    )

    model, tokenizer = load_summarization_model_and_tokenizer()

    xsum_data = XsumDataset(datasets.load_dataset("xsum")["test"])
    total = len(xsum_data)
    print(str(total) + " total documents to summarize")
    i = 0
    for x in xsum_data:
        i += 1
        gen_summary = generate_summaries(model, tokenizer, x["document"])

        store_model_summaries(
            "xsum",
            model.config.name_or_path,
            model.config.to_dict(),
            {x["id"]: gen_summary},
        )
        if i % 10 == 0:
            print(str(i) + " of " + str(total) + " summaries completed.")
