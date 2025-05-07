import streamlit as st
from navigation import NavigationManager
from utils import get_feature_description

# Define the feature extraction methods
FEATURE_METHODS = ["Raw Pixel Values", "Matrix Properties (Pixel,Rank,Det,Trace)","Matrix Properties (Rank,Det,Trace)","Pixels and MPs of Scalograms(CWT,STFT)","MPs of Original,CWT,STFT","Pixels of Original and MPs of Original,CWT,STFT","Pixels of Original,CWT,STFT and MPs of Original,CWT,STFT","Smith Normal Form With Window Size 5"]
CLASSIFICATION_METHODS=["CNP","CP","CN","NP"]

def show_feature_selection_page():
    """Display the page for selecting feature extraction method"""
    st.markdown("<h1 class='main-header'>Select Feature Extraction Method</h1>", unsafe_allow_html=True)
    
    # Navigation bar
    nav = NavigationManager()
    nav.display_page_navigation()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Feature Extraction Methods</h2>", unsafe_allow_html=True)
                
        for method in FEATURE_METHODS:
            if st.button(f"Select: {method}", key=f"select_{method.replace(' ', '_')}", use_container_width=True):
                st.session_state.selected_feature = method
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Feature Method Details</h2>", unsafe_allow_html=True)
        for method in CLASSIFICATION_METHODS:
            if st.button(f"Select: {method}", key=f"select_{method.replace(' ', '_')}", use_container_width=True):
                st.session_state.selected_classification = method
                st.rerun()
        
        # if st.session_state.get("selected_feature") == "Raw Pixel Values":
        #     st.markdown("### Raw Pixel Values")
            
        #     # Display description of the pixel values method
        #     description = """
        #     <p><strong>Raw Pixel Values</strong> is the simplest feature extraction method. It uses the direct 
        #     pixel intensity values from the image as features.</p>
            
        #     <p>This approach:</p>
        #     <ul>
        #         <li>Preserves all original information from the image</li>
        #         <li>Requires no complex transformations</li>
        #         <li>Works well with deep learning models that can learn patterns directly from raw data</li>
        #         <li>May require normalization (scaling pixel values between 0-1) for better results</li>
        #     </ul>
            
        #     <p>While simple, this method can be effective for many image analysis tasks, especially 
        #     when combined with appropriate machine learning algorithms.</p>
        #     """
        #     st.markdown(description, unsafe_allow_html=True)
            
        #     # Show current selection and confirmation
        #     st.success("You have selected: **Raw Pixel Values**")
            
        #     # Show a sample visualization
            
            
        # elif st.session_state.get("selected_feature") == "Matrix Properties (Pixel,Rank,Det,Trace)":
        #     st.markdown("### Matrix Properties (Pixel,Rank,Det,Trace)")
            
        #     # Display description of the matrix properties method
        #     description = """
        #     <p><strong>Matrix Properties</strong> extracts multiple mathematical properties from the image when treated as a matrix:</p>
            
        #     <p>This method:</p>
        #     <ul>
        #         <li>Resizes the image to 32×32 pixels</li>
        #         <li>Converts the image to grayscale</li>
        #         <li>Extracts the following properties:</li>
        #         <ul>
        #             <li>Pixel values: The intensity value of each pixel in the grayscale image</li>
        #             <li>Rank: The rank of the image matrix (number of linearly independent rows/columns)</li>
        #             <li>Determinant: The determinant of the image matrix</li>
        #             <li>Trace: The sum of the elements on the main diagonal of the matrix</li>
        #         </ul>
        #     </ul>
            
        #     <p>This method captures both the raw image data and important algebraic properties of the image 
        #     when represented as a matrix, providing rich features for classification tasks.</p>
        #     """
        #     st.markdown(description, unsafe_allow_html=True)
            
        #     # Show current selection and confirmation
        #     st.success("You have selected: **Matrix Properties (Pixel,Rank,Det,Trace)**")
            
        #     # Show a sample visualization
        #     st.image("https://via.placeholder.com/400x300?text=Matrix+Properties", 
        #              caption="Sample visualization of Matrix Properties analysis", 
        #              use_container_width=True)
            
        # elif st.session_state.get("selected_feature") == "Matrix Properties (Rank,Det,Trace)":
        #     st.markdown("### Matrix Properties (Rank,Det,Trace)")
            
        #     # Display description of the matrix properties method
        #     description = """
        #     <p><strong>Matrix Properties</strong> extracts multiple mathematical properties from the image when treated as a matrix:</p>
            
        #     <p>This method:</p>
        #     <ul>
        #         <li>Resizes the image to 32×32 pixels</li>
        #         <li>Converts the image to grayscale</li>
        #         <li>Extracts the following properties:</li>
        #         <ul>
        #             <li>Pixel values: The intensity value of each pixel in the grayscale image</li>
        #             <li>Rank: The rank of the image matrix (number of linearly independent rows/columns)</li>
        #             <li>Determinant: The determinant of the image matrix</li>
        #             <li>Trace: The sum of the elements on the main diagonal of the matrix</li>
        #         </ul>
        #     </ul>
            
        #     <p>This method captures both the raw image data and important algebraic properties of the image 
        #     when represented as a matrix, providing rich features for classification tasks.</p>
        #     """
        #     st.markdown(description, unsafe_allow_html=True)
            
        #     # Show current selection and confirmation
        #     st.success("You have selected: **Matrix Properties (Rank,Det,Trace)**")
            
        #     # Show a sample visualization
        #     st.image("https://via.placeholder.com/400x300?text=Matrix+Properties", 
        #              caption="Sample visualization of Matrix Properties analysis", 
        #              use_container_width=True)
            
        # elif st.session_state.get("selected_feature") == "Pixels and MPs of Scalograms(CWT,STFT)":
        #     st.markdown("### Pixels and MPs of Scalograms(CWT,STFT)")
            
        #     # Display description of the matrix properties method
        #     description = """
        #     <p><strong>Matrix Properties</strong> extracts multiple mathematical properties from the image when treated as a matrix:</p>
            
        #     <p>This method:</p>
        #     <ul>
        #         <li>Resizes the image to 32×32 pixels</li>
        #         <li>Converts the image to grayscale</li>
        #         <li>Extracts the following properties:</li>
        #         <ul>
        #             <li>Pixel values: The intensity value of each pixel in the grayscale image</li>
        #             <li>Rank: The rank of the image matrix (number of linearly independent rows/columns)</li>
        #             <li>Determinant: The determinant of the image matrix</li>
        #             <li>Trace: The sum of the elements on the main diagonal of the matrix</li>
        #         </ul>
        #     </ul>
            
        #     <p>This method captures both the raw image data and important algebraic properties of the image 
        #     when represented as a matrix, providing rich features for classification tasks.</p>
        #     """
        #     st.markdown(description, unsafe_allow_html=True)
            
        #     # Show current selection and confirmation
        #     st.success("You have selected: **Pixels and MPs of Scalograms(CWT,STFT)**")
            
        #     # Show a sample visualization
        #     st.image("https://via.placeholder.com/400x300?text=Matrix+Properties", 
        #              caption="Sample visualization of Matrix Properties analysis", 
        #              use_container_width=True)
            
        # else:
            # st.info("Please select a feature extraction method from the left panel.")
            
            # # General information about feature methods
            # st.markdown("""
            # ### About Feature Extraction Methods
            
            # We offer two approaches for extracting features from X-ray images:
            
            # - **Raw Pixel Values:** Uses the direct intensity values from the image
            # - **Matrix Properties:** Treats the image as a matrix and extracts mathematical properties
            
            # Each method has different strengths and may perform better for different diagnostic tasks.
            # """)
        
        # Provide option to change selection
        if st.session_state.get("selected_feature"):
            if st.button("Clear Selection"):
                st.session_state.selected_feature = None
                st.rerun()
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Method performance section
        # if st.session_state.get("selected_feature") == "Raw Pixel Values":
        #     st.markdown("<div class='card'>", unsafe_allow_html=True)
        #     st.markdown("<h3>Method Performance</h3>", unsafe_allow_html=True)
            
        #     # Performance metrics
        #     st.markdown("""
        #     **Performance metrics for Raw Pixel Values:**
            
        #     - **Accuracy:** 84.2%
        #     - **Sensitivity:** 83.5%
        #     - **Specificity:** 85.0%
        #     - **Processing Time:** Very Fast
            
        #     This method works well as a baseline approach and for:
        #     - Simple classification tasks
        #     - Deep learning applications
        #     - Quick prototype development
        #     """)
            
        #     st.markdown("</div>", unsafe_allow_html=True)
            
        # elif st.session_state.get("selected_feature") == "Matrix Properties (Pixel,Rank,Det,Trace)":
        #     st.markdown("<div class='card'>", unsafe_allow_html=True)
        #     st.markdown("<h3>Method Performance</h3>", unsafe_allow_html=True)
            
        #     # Performance metrics
        #     st.markdown("""
        #     **Performance metrics for Matrix Properties:**
            
        #     - **Accuracy:** 86.7%
        #     - **Sensitivity:** 85.2%
        #     - **Specificity:** 88.1%
        #     - **Processing Time:** Fast
            
        #     This method performs well for:
        #     - Detecting structural abnormalities
        #     - Pattern recognition tasks
        #     - Cases where mathematical properties of the image provide important diagnostic insights
        #     """)
            
        #     st.markdown("</div>", unsafe_allow_html=True)

        # elif st.session_state.get("selected_feature") == "Matrix Properties (Rank,Det,Trace)":
        #     st.markdown("<div class='card'>", unsafe_allow_html=True)
        #     st.markdown("<h3>Method Performance</h3>", unsafe_allow_html=True)
            
        #     # Performance metrics
        #     st.markdown("""
        #     **Performance metrics for Matrix Properties:**
            
        #     - **Accuracy:** 86.7%
        #     - **Sensitivity:** 85.2%
        #     - **Specificity:** 88.1%
        #     - **Processing Time:** Fast
            
        #     This method performs well for:
        #     - Detecting structural abnormalities
        #     - Pattern recognition tasks
        #     - Cases where mathematical properties of the image provide important diagnostic insights
        #     """)
            
        #     st.markdown("</div>", unsafe_allow_html=True)

        # elif st.session_state.get("selected_feature") == "Pixels and MPs of Scalograms(CWT,STFT)":
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            # st.markdown("<h3>Method Performance</h3>", unsafe_allow_html=True)
            
            # # Performance metrics
            # st.markdown("""
            # **Performance metrics for Pixels and MPs of Scalograms(CWT,STFT):**
            
            # - **Accuracy:** 86.7%
            # - **Sensitivity:** 85.2%
            # - **Specificity:** 88.1%
            # - **Processing Time:** Fast
            
            # This method performs well for:
            # - Detecting structural abnormalities
            # - Pattern recognition tasks
            # - Cases where mathematical properties of the image provide important diagnostic insights
            # """)
            
            # st.markdown("</div>", unsafe_allow_html=True)