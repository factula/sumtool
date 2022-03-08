# sumtool
A toolkit for understanding factuality & consistency errors in summarization models.

### Core Features
- A harness for generating text summaries with automated factuality evaluations 
- - NLI (textual entailment)
- - Question answering
- - _Other metrics (BERT-Score, Rouge Score, etc.)_

- An interactive query interface for exploring generated summaries (i.e. XSum or custom dataset)
- - Search for common factuality errors across your dataset (i.e. find all numerical errors)
- - Explore faithfulness & factuality annotations (if available)

## References

[References](reference.md)

## Setup
Setup (python 3.8):
```
pip install -r requirements.txt
pip install .
```

### Run Streamlit app
```
streamlit run interface/app.py
```

You can also run interfaces individually, i.e. 
```
streamlit run interface/summary_interface.py
```

### Contributors

Setup (python 3.8):
```
pip install -r requirements.dev.txt
pip install -Ue .
```

Before commiting:
```
black sumtool/ interface/ scripts/
flake8 sumtool/ interface/ scripts/
```

### Run on Google Colab for GPU

1. Create a Github token to access your private repositories. Follow these steps here:
[Github: Creating a Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

2. Create a new Colab notebook and set the runtime type to GPU

3. Add the following commands in the first cell to clone the repository and install the requirements
```
!git clone https://[your-git-token]@github.com/cs6741/summary-analysis.git
!pip install -r /content/summary-analysis/requirements.txt
```

4. Add the following command to run the text generation script
```
!python /content/generate_xsum_summary.py --bbc_ids [idx1,idx2] --data_split [train|test]
```

### Storage documentation

**Pipeline for storage:**
1. Store generated summaries
   - by generating them using a custom model ([example](sumtool/predict_xsum_summary.py))
   - by loading them from an external dataset/paper ([example](scripts/store_xsum_annotated.py))
2. Compute summary metrics for stored summaries using sumtool.

#### `/data/<dataset>/<model-i>-summaries.json`
	<document_id>: 
		summary: the generated summary,
		metadata: ...metadata for the generated summary, i.e. annotations / score / entropy
	
      
#### `/data/<dataset>/<model-id>-metrics.json`
	<document_id>: 
		...metrics for a stored summary, i.e. rouge-score, bert-score
