import streamlit as st
import requests
from typing import Optional

def init_auth():
    """Initialize authentication state in session."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None

def handle_callback():
    """Handle the OpenShift OAuth callback."""
    if "access_token" in st.query_params:
        token = st.query_params["access_token"]
        # Get user info from OpenShift OAuth
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://openshift.default.svc/apis/user.openshift.io/v1/users/~", 
                              headers=headers, 
                              verify=False)  # Skip SSL verification for internal OpenShift API
        
        if response.status_code == 200:
            user_info = response.json()
            st.session_state.authenticated = True
            st.session_state.user_info = user_info
            # Clear the token from URL
            st.query_params.clear()
            st.rerun()

def logout():
    """Handle the logout process."""
    if st.session_state.authenticated:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.rerun()

def require_auth():
    """Decorator to require authentication for a page."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            init_auth()
            handle_callback()
            
            if not st.session_state.authenticated:
                st.warning("Please log in to access this page.")
                return
            
            return func(*args, **kwargs)
        return wrapper
    return decorator 