import streamlit as st
from navigation import NavigationManager
from data_manager import load_model, extract_features, predict_disease, generate_report
import time

def show_processing_page():
    """Display the page for processing the X-ray image with selected features"""
    st.markdown("<h1 class='main-header'>Processing X-ray Image</h1>", unsafe_allow_html=True)
    
    # Navigation bar
    nav = NavigationManager()
    nav.display_page_navigation()
    
    # Main content
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h2 class='sub-header'>Processing with {st.session_state.selected_feature}</h2>", unsafe_allow_html=True)
    
    # Process the image if not already processed
    if st.session_state.prediction_results is None:
        # Create a progress tracker
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Display the original image
        st.markdown("### Original X-ray Image")
        st.image(st.session_state.uploaded_image, width=400)
        status_text.text("Loading the selected model...")
        progress_bar.progress(10)
        time.sleep(0.5)  # Simulate processing time
        
        # Step 2: Load model
        model = load_model(st.session_state.selected_feature,st.session_state.selected_classification)
        progress_bar.progress(30)
        status_text.text("Extracting features from the image...")
        time.sleep(0.8)  # Simulate processing time
        
        if model:
            # Step 3: Extract features
            features, processing_steps = extract_features(
                st.session_state.uploaded_image, 
                st.session_state.selected_feature
            )
            
            # Display processing steps
            st.markdown("### Processing Steps")
            for i, (step_name, step_image) in enumerate(processing_steps):
                st.markdown(f"**Step {i+1}: {step_name}**")
                
                # Check if it's a figure or an image array
                if hasattr(step_image, 'savefig'):  # It's a matplotlib figure
                    st.pyplot(step_image)
                else:  # It's an image array
                    st.image(step_image, width=400)
                
                # Update progress
                progress_value = 30 + (i+1) * 40 // len(processing_steps)
                progress_bar.progress(progress_value)
                status_text.text(f"Processing step {i+1} of {len(processing_steps)}: {step_name}")
                time.sleep(0.5)  # Simulate processing time
            
            # Save processing visualizations for results page
            st.session_state.processing_visualizations.extend(processing_steps)
            
            # Step 4: Make predictions
            status_text.text("Generating predictions...")
            progress_bar.progress(80)
            time.sleep(0.8)  # Simulate processing time
            
            predictions = predict_disease(features, model)
            st.session_state.prediction_results = predictions
            
            # Step 5: Generate report
            status_text.text("Creating diagnostic report...")
            progress_bar.progress(90)
            time.sleep(0.7)  # Simulate processing time
            
            report = generate_report(
                st.session_state.patient_info,
                predictions,
                st.session_state.selected_feature
            )
            st.session_state.report = report
            
            # Complete
            progress_bar.progress(100)
            status_text.text("Processing complete! Redirecting to results page...")
            time.sleep(1)  # Pause before redirect
            
            # Automatically redirect to results
            st.success("Analysis completed successfully!")
            nav.go_to_page('results')
            st.rerun()
        else:
            st.error("Failed to load the selected model. Please try again or select a different feature extraction method.")
            progress_bar.progress(0)
            status_text.text("Processing failed. Please try again.")
    else:
        # If already processed, show a message and redirect
        st.info("Processing already completed. Redirecting to results...")
        nav.go_to_page('results')
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add technical details expander
    with st.expander("Technical Details"):
        st.markdown("""
        ### Processing Pipeline
        
        1. **Image Preprocessing**
           - Normalization
           - Noise reduction
           - Contrast enhancement
        
        2. **Feature Extraction**
           - Application of selected method
           - Generation of feature vector
        
        3. **Model Inference**
           - Prediction using pre-trained model
           - Confidence scoring
        
        4. **Report Generation**
           - Integration of patient data
           - Formatting results for clinical use
        """)