import streamlit as st

# Custom CSS styles for the application
CUSTOM_CSS = """
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background-color: #ff4b4b;
        color: white;
        padding: 10px 10px;
        border-radius: 25px;
    }
    .header-container {
        text-align: center;
        color: white;
    }
    .error-message {
        color: #ff0000;
        display: inline-block;
        margin-left: 10px;
        vertical-align: middle;
        min-width: 150px;
    }
    .button-container {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .big-font {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    .result-area {
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
    }
    
    /* Messenger-style chat bubbles */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin: 15px 0;
    }
    .question-bubble {
        align-self: flex-end;
        background-color: #0084ff;
        color: white;
        padding: 10px 15px;
        border-radius: 18px;
        max-width: 80%;
        margin-left: 20%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .answer-bubble {
        align-self: flex-start;
        background-color: #f0f2f6;
        color: black;
        padding: 10px 15px;
        border-radius: 18px;
        max-width: 80%;
        margin-right: 20%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
"""

def apply_custom_css():
    """Apply the custom CSS styles to the Streamlit app"""
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

def display_header(title):
    """Display the header with a custom title"""
    st.markdown(
        f'<div class="header-container"><h1 class="main-header">{title}</h1></div>', 
        unsafe_allow_html=True
    )

    st.markdown("---")

def display_footer(text):
    """Add a footer with information"""
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666666; font-size: 0.8em;'>
        {text} • Powered by FSI
        </div>
        """,
        unsafe_allow_html=True
    )