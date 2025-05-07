import streamlit as st

class NavigationManager:
    """Manage navigation between pages in the application"""
    
    def __init__(self):
        self.pages = ['welcome', 'upload', 'feature_selection', 'processing', 'results']
        self.page_names = ['Welcome', 'Upload Image', 'Feature Selection', 'Processing', 'Results']
    
    def go_to_page(self, page):
        """Change the current page in session state"""
        st.session_state.current_page = page
        
        # Reset relevant session state variables when going back to welcome
        if page == 'welcome':
            from data_manager import reset_analysis
            reset_analysis()
    
    def display_navigation_sidebar(self):
        """Display navigation in the sidebar with visual indication of current step"""
        st.write("### Process Steps")
        
        current_idx = self.pages.index(st.session_state.current_page)
        
        for i, (page, name) in enumerate(zip(self.pages, self.page_names)):
            # Determine the status of this step
            if i < current_idx:
                status = "complete"
                prefix = "✅ "
            elif i == current_idx:
                status = "current"
                prefix = "➡️ "
            else:
                status = "incomplete"
                prefix = "⬜ "
            
            # Display step with appropriate styling
            st.markdown(f"<div class='step-{status}'>{prefix} {name}</div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation buttons based on current page
        if st.session_state.current_page != 'welcome':
            if st.button("↩️ Reset & Start Over"):
                self.go_to_page('welcome')
        
        # Help information
        with st.expander("ℹ️ Need Help?"):
            st.write("""
            **How to use this application:**
            
            1. Upload a chest X-ray image
            2. Select a feature extraction method
            3. Review the analysis process
            4. Get a detailed diagnostic report
                     
            If you encounter any issues, please contact support.
            """)

    def display_page_navigation(self):
        """Display in-page navigation buttons"""
        st.markdown("<div class='navigation'>", unsafe_allow_html=True)
        
        current_idx = self.pages.index(st.session_state.current_page)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # Back button if not on first page
        if current_idx > 0:
            with col1:
                prev_page = self.pages[current_idx - 1]
                prev_name = self.page_names[current_idx - 1] 
                if st.button(f"← Back to {prev_name}", key=f"back_to_{prev_page}"):
                    self.go_to_page(prev_page)
        
        # Next button if not on last page and conditions are met
        if current_idx < len(self.pages) - 1:
            with col3:
                next_page = self.pages[current_idx + 1]
                next_name = self.page_names[current_idx + 1]
                
                # Check conditions for enabling the next button
                can_proceed = False
                
                if current_idx == 0:  # From welcome to upload
                    can_proceed = True
                elif current_idx == 1:  # From upload to feature selection
                    can_proceed = st.session_state.uploaded_image is not None
                elif current_idx == 2:  # From feature selection to processing
                    can_proceed = st.session_state.selected_feature is not None
                elif current_idx == 3:  # From processing to results
                    can_proceed = st.session_state.prediction_results is not None
                
                if can_proceed:
                    if st.button(f"Continue to {next_name} →", key=f"go_to_{next_page}"):
                        self.go_to_page(next_page)
        
        st.markdown("</div>", unsafe_allow_html=True)