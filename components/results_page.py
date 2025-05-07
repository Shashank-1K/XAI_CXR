import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from navigation import NavigationManager
from utils import create_visualization,svd_visualisation,Pattern_Visualization

def show_results_page():
    """Display the results page with prediction results and visualizations"""
    st.markdown("<h1 class='main-header'>Analysis Results</h1>", unsafe_allow_html=True)
    
    # Navigation bar
    nav = NavigationManager()
    nav.display_page_navigation()
    
    # Check if we have results to display
    if st.session_state.prediction_results is None:
        st.warning("No analysis results available. Please complete the analysis process first.")
        if st.button("Go to Upload Page"):
            nav.go_to_page('upload')
        return
    
    # Tabs for results presentation
    tab1, tab2, tab3, tab4, tab5 ,tab6= st.tabs(["Report", "Processing Steps", "Visualization", "Raw Data","SVD Visualisation","Pattern Visualization"])
    
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Diagnostic Report</h2>", unsafe_allow_html=True)
        
        report = st.session_state.report
        patient = report["patient_info"]
        analysis = report["analysis"]
        
        # Create two columns for patient info and findings
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Patient Information")
            st.markdown(f"""
            - **Name:** {patient['name']}
            - **Age:** {patient['age']}
            - **Gender:** {patient['gender']}
            - **Exam Date:** {patient['exam_date']}
            """)
        
        with col2:
            st.markdown("### Analysis Method")
            st.markdown(f"**Feature Extraction:** {analysis['method']}")
            
            st.markdown("### Primary Findings")
            primary_confidence = float(analysis['primary_confidence'].strip('%')) / 100
            differential_confidence = float(analysis['differential_confidence'].strip('%')) / 100
            
            # Use color coding based on confidence
            primary_color = "green" if st.session_state.label[0] == "Normal" else "orange"
            
            st.markdown(f"""
            - **Primary Diagnosis:** <span style='color:{primary_color};font-weight:bold;font-size:25px;'>{analysis['primary_finding']}</span> (Confidence: {analysis['primary_confidence']})
            - **Differential Diagnosis:** {analysis['differential_diagnosis']} (Confidence: {analysis['differential_confidence']})
            """, unsafe_allow_html=True)
        
        # Recommendations section
        st.markdown("### Recommendations")
        st.markdown(f"<div class='info-box' >{analysis['recommendations']}</div>", unsafe_allow_html=True)
        
        # # Add option to download report
        # col1, col2 = st.columns([1, 1])
        # with col1:
        #     if st.button("Download Report as PDF", key="download_report"):
        #         st.info("In a real application, this would generate and download a PDF report.")
        # with col2:
        #     if st.button("Email Report", key="email_report"):
        #         st.info("In a real application, this would send the report via email.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Processing Steps</h2>", unsafe_allow_html=True)
        
        # Display processing steps and visualizations
        steps = st.session_state.processing_visualizations
        
        # Original image
        st.subheader("Original X-ray Image")
        st.image(st.session_state.uploaded_image, width=400)
        
        # Intermediate steps
        for title, img in steps:
            st.subheader(title)
            
            # Check if it's a figure or an image array
            if hasattr(img, 'savefig'):  # It's a matplotlib figure
                st.pyplot(img)
            else:  # It's an image array
                st.image(img, width=400)
                
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Visualization</h2>", unsafe_allow_html=True)
        
        # Get prediction probabilities
        predictions = st.session_state.prediction_results
        
        # Create visualization
        fig = create_visualization(predictions)
        st.pyplot(fig)
        
        # Add explanation for the visualizations
        st.markdown("""
        ### Understanding These Visualizations
        
        - **Bar Chart**: Shows the confidence score of each potential diagnosis, with higher values indicating stronger likelihood.
        
        These visualizations help to explain the AI system's reasoning process and focus areas for diagnosis.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Raw Data</h2>", unsafe_allow_html=True)
        
        # Display raw prediction data
        st.subheader("Prediction Probabilities")
        
        # Create a DataFrame for better display
        df = pd.DataFrame({
            'Disease': [p[0] for p in st.session_state.prediction_results],
            'Probability': [p[1] for p in st.session_state.prediction_results]
        })
        
        # Format probability as percentage
        df['Probability'] = df['Probability'].map('{:.2%}'.format)
        
        st.dataframe(df, use_container_width=True)
        
        # Additional information about the model
        st.markdown("### Model Information")
        st.markdown(f"""
        - **Method:** {st.session_state.selected_feature}
        - **Model Version:** 1.0
        - **Last Updated:** 2025-03-15
        - **Training Data:** 50,000 annotated chest X-rays
        """)
        
        # # Option to download raw data
        # if st.button("Download Raw Data", key="download_data"):
        #     st.info("In a real application, this would download the raw prediction data as CSV.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    with tab5:
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(svd_visualisation(st.session_state.uploaded_image), width=400)
    with tab6:
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(Pattern_Visualization(st.session_state.uploaded_image,), width=400)
 
    


    
    # Add a "Start New Analysis" button at the bottom
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("Start New Analysis", key="new_analysis", use_container_width=True):
            nav.go_to_page('welcome')
    st.markdown("</div>", unsafe_allow_html=True)
