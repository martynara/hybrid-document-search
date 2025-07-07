import streamlit as st
import requests
import base64
from _style import *

def display_content():
    st.image("pages/img/generative-AI.png", use_container_width=True)

def main():
    apply_custom_css()
    display_header("Aplikacje Generative AI")
    display_content()
    display_footer("Wyszukuje hybrydowo")

if __name__ == "__main__":
    main()

