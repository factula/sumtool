import streamlit as st
from datasets import load_dataset

# Load datasets
# TODO: use solution from https://github.com/cs6741/summary-analysis/issues/2 to align data
data_xsum = load_dataset("xsum")
data_xsum_factuality = load_dataset("xsum_factuality")

# UI
st.header("CS6741 - Summary Analysis")

data_split = st.selectbox("Select data split", options=["train", "test", "validation"])

col1, col2 = st.columns(2)

idx_xsum = col1.number_input("Select data index", value=0)
col1.subheader("XSUM")
col1.write(data_xsum[data_split][idx_xsum])

idx_xsum_factuality = col2.number_input("Select data index (factuality)", value=0)
col2.subheader("XSUM Factuality")
col2.write(data_xsum_factuality[data_split][idx_xsum_factuality])

foo = {1 : 2}
