import streamlit as st
from navigation import NavigationManager

def show_welcome_page():
    """Display the welcome page with introduction to the application"""
    st.markdown("<h1 class='main-header'>ArogyaVignanX</h1>", unsafe_allow_html=True)
    
    # Navigation buttons
    nav = NavigationManager()
    nav.display_page_navigation()
    
    # Main content with two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Welcome to the Advanced Diagnostic Assistant</h2>", unsafe_allow_html=True)
        st.markdown("""
        This system uses state-of-the-art machine learning techniques to analyze chest X-ray images 
        and assist in the diagnosis of various pulmonary conditions.
        
        <div class='info-box' style='color: black;'>
        <strong>How it works:</strong><br>
        1. Upload a chest X-ray image<br>
        2. Select a feature extraction method<br>
        3. Review the analysis process<br>
        4. Get a detailed diagnostic report
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p>Click the button below to begin your analysis:</p>", unsafe_allow_html=True)
        if st.button("Start New Analysis", key="start_btn", use_container_width=True):
            nav.go_to_page('upload')
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box' style='color:black;'>
        <strong>System capabilities:</strong><br>
        • different feature extraction methods<br>
        • Detailed visual explanations<br>
        • Comprehensive diagnostic reports<br>
        • Supporting clinical decision-making
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional information section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>About This System</h2>", unsafe_allow_html=True)
    
    # tab1, tab2, tab3 = st.tabs(["Features", "Technology", "Usage Guidelines"])
    tab1, tab2 = st.tabs(["Features", "Usage Guidelines"])
    with tab1:
        st.markdown("""
        ### Key Features
        
        - **Multi-Disease Classification**: Capable of identifying 3 common pulmonary conditions
        - **Diverse Feature Extraction**: Various methods to optimize for different conditions
        - **Visual Explanations**: Step-by-step visualization of the analysis process
        - **Detailed Reports**: Comprehensive diagnostic reports with recommendations
        - **User-Friendly Interface**: Intuitive design for clinical environments
        """)
    
    # with tab2:
    #     st.markdown("""
    #     ### Technology
        
    #     This system leverages state-of-the-art machine learning techniques including:
        
    #     - **Advanced Feature Extraction**: From traditional methods to deep learning approaches
    #     - **Ensemble Learning**: Multiple models for improved accuracy
    #     - **Visualization Tools**: For better interpretation of results
    #     - **Streamlit Framework**: For an interactive, responsive user interface
    #     """)
    
    with tab2:
        st.markdown("""
        ### Usage Guidelines
        
        - This system is designed as a diagnostic aid, not a replacement for clinical judgment
        - Always verify results with standard clinical protocols
        - For optimal results, use high-quality X-ray images
        - The system performs best with PA (posteroanterior) chest X-rays
        - Results should be interpreted by qualified healthcare professionals
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("<div class='footer'>© 2025 X-ray Analysis System | For Clinical Use Only</div>", unsafe_allow_html=True)