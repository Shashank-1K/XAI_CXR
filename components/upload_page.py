import streamlit as st
from PIL import Image
import numpy as np
from utils import display_image_info
from navigation import NavigationManager
import datetime

def show_upload_page():
    """Display the page for uploading X-ray images"""
    st.markdown("<h1 class='main-header'>Upload X-ray Image</h1>", unsafe_allow_html=True)
    
    # Navigation bar
    nav = NavigationManager()
    nav.display_page_navigation()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Upload Your Chest X-ray Image</h2>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Store the uploaded image in session state
                st.session_state.uploaded_image = image_array
                
                st.success("Image successfully uploaded!")
                
                # Patient information
                st.markdown("<h3>Patient Information</h3>", unsafe_allow_html=True)
                patient_name = st.text_input("Patient Name", "John Doe")
                
                # col1, col2 = st.columns(2)
                # with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=45)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                
                    
                
                # Exam date
                exam_date = st.date_input(
                    "Exam Date", 
                    value=datetime.datetime.now(),
                    max_value=datetime.datetime.now()
                )
                
                # Store patient info in session state
                st.session_state.patient_info = {
                    "name": patient_name,
                    "age": age,
                    "gender": gender
                }
                st.session_state.exam_date = exam_date.strftime("%Y-%m-%d")
                
            except Exception as e:
                st.error(f"Error: {e}. Please upload a valid image file.")
        
        else:
            st.info("Please upload a chest X-ray image to proceed.")
            
            # Sample images option
            st.markdown("---")
            
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Image Preview</h2>", unsafe_allow_html=True)
        
        if st.session_state.uploaded_image is not None:
            st.image(st.session_state.uploaded_image, caption="Uploaded X-ray Image", use_container_width=True)
            
            # Display image information
            info = display_image_info(st.session_state.uploaded_image)
            st.markdown(info, unsafe_allow_html=True)
            
          
            
            
            
        else:
            st.info("Please upload an image to see the preview here.")
            
            # Show placeholder with guidelines
            
            
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer with helpful information
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Preparing Your X-ray Images</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    To ensure the best possible diagnostic accuracy, please follow these guidelines:
    
    1. **Image Format**: JPEG or PNG files are preferred
    2. **Resolution**: At least 32x32 pixels for optimal feature extraction
    3. **Positioning**: The entire chest should be visible with minimal rotation
    4. **Processing**: Avoid applying filters or enhancements to the original X-ray
    
    The system works best with standard PA (posteroanterior) chest X-rays taken with the patient standing upright.
    """)
    st.markdown("</div>", unsafe_allow_html=True)