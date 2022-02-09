# sumtool

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

### Run on Google Colab for GPU

Steps to run this program on Colab:


1. Create a Github token to access your private repositories. Follow these steps here:
[Github: Creating a Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

2. Create a new Colab notebook and set the runtime type to GPU
3. Add the following commands in the first cell to clone the repository and install the requirements
```
!git clone https://[your-git-token]@github.com/cs6741/summary-analysis.git
!pip install -r /content/summary-analysis/requirements.txt
```
3. Add the following command to run the predictions script
```
!python /content/predict_xsum_summary.py --data_index [some-data-index] --data_split [train|test]
```