import streamlit as st
from PIL import Image
import numpy as np
import time
import pandas as pd

# Import modules
from utils import setup_page, load_css
from navigation import NavigationManager
from components.welcome_page import show_welcome_page
from components.upload_page import show_upload_page
from components.feature_selection_page import show_feature_selection_page
from components.processing_page import show_processing_page
from components.results_page import show_results_page
from data_manager import initialize_session_state

# Main application function
def main():
    # Setup page configuration
    setup_page()
    
    # Load custom CSS
    load_css()
    
    # Initialize session state if needed
    initialize_session_state()
    
    # Create sidebar navigation
    with st.sidebar:
        
        st.title("Navigation")
        
        # Display navigation based on current page
        nav = NavigationManager()
        nav.display_navigation_sidebar()
    
    # Show appropriate page based on session state
    if st.session_state.current_page == 'welcome':
        show_welcome_page()
    elif st.session_state.current_page == 'upload':
        show_upload_page()
    elif st.session_state.current_page == 'feature_selection':
        show_feature_selection_page()
    elif st.session_state.current_page == 'processing':
        show_processing_page()
    elif st.session_state.current_page == 'results':
        show_results_page()

if __name__ == "__main__":
    main()