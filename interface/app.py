import streamlit as st
from factuality_interface import render_factuality_interface
from faithfulness_interface import render_faithfulness_interface
from model_interface import render_model_interface

# Ngram visualization - slow, commented out for now
# from ngram_interface import render_ngram_interface


def render():
    pages = {
        "Factuality Annotations": render_factuality_interface,
        "Faithfulness Annotations": render_faithfulness_interface,
        "Model Interface": render_model_interface,
        # "Ngram Interface": render_ngram_interface
    }

    st.sidebar.title("CS6741 - Summary Analysis")
    selected_page = st.sidebar.radio("Select a page", options=list(pages.keys()))

    pages[selected_page]()


if __name__ == "__main__":
    render()
