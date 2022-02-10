# sumtool
A toolkit for understanding factuality & consistency errors in summarization models.

### Core Features
- A harness for generating text summaries with automated consistency & factuality checks 
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
```

### Run Streamlit app
```
streamlit run interface/app.py
```

You can also run interfaces individually, i.e. 
```
streamlit run interface/factuality_interface.py
```

### Contributors

Setup (python 3.8):
```
pip install -r requirements.dev.txt
```

Before commiting:
```
black sumtool/ interface/
flake8 sumtool/ interface/
```

