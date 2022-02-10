import streamlit as st
from summary_interface import render_summary_interface
from model_interface import render_model_interface


def render():
    pages = {
        "Explore Summarization Datasets": render_summary_interface,
        "Model Interface": render_model_interface,
    }

    st.sidebar.title("CS6741 - Summary Analysis")
    selected_page = st.sidebar.radio("Select a page", options=list(pages.keys()))

    pages[selected_page]()


if __name__ == "__main__":
    render()
