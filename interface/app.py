import streamlit as st
from factuality_interface import render_factuality_interface
from model_interface import render_model_interface


def render():
    pages = {
        "Factuality Annotations": render_factuality_interface,
        "Model Interface": render_model_interface,
    }

    st.sidebar.title("CS6741 - Summary Analysis")
    selected_page = st.sidebar.radio("Select a page", options=list(pages.keys()))

    pages[selected_page]()


if __name__ == "__main__":
    render()
