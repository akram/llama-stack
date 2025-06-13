import streamlit as st
import requests
import base64
from urllib.parse import urlencode

def init_auth():
    """Initialize authentication state in session."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "access_token" not in st.session_state:
        st.session_state.access_token = None

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
            st.session_state.access_token = None
            st.rerun()

# Constants
CLIENT_ID = "lsplayground"
CLIENT_SECRET = "your-client-secret"  # Replace with actual value
REDIRECT_URI = "https://playground-llamastack-kserve-llama3.apps.rosa.akram.vsil.p3.openshiftapps.com/callback"
OAUTH_HOST = "https://oauth.akram.vsil.p3.openshiftapps.com"
AUTHORIZE_URL = f"{OAUTH_HOST}/oauth/authorize"
TOKEN_URL = f"{OAUTH_HOST}/oauth/token"

def require_auth():
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Initialize session state
            init_auth()
            
            # Handle callback: code exchange
            code = st.query_params.get("code")

            if not st.session_state.authenticated:
                if code:
                    try:
                        # Debugging info
                        st.write("Debug - Request details:")
                        st.write(f"Code: {code}")
                        st.write(f"Redirect URI: {REDIRECT_URI}")

                        # Encode client_id and secret in Basic Auth
                        credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
                        b64_credentials = base64.b64encode(credentials.encode()).decode()

                        headers = {
                            "Authorization": f"Basic {b64_credentials}",
                            "Content-Type": "application/x-www-form-urlencoded",
                            "Accept": "application/json"
                        }

                        data = {
                            "grant_type": "authorization_code",
                            "code": code,
                            "redirect_uri": REDIRECT_URI,
                            "scope": "user:info user:check-access"
                        }

                        response = requests.post(
                            TOKEN_URL,
                            data=data,
                            headers=headers,
                            timeout=10
                        )

                        # Debugging
                        st.write("Debug - Response status:", response.status_code)
                        st.write("Debug - Response headers:", response.headers)
                        st.write("Debug - Response body:", response.text)

                        response.raise_for_status()
                        token_data = response.json()
                        st.session_state.access_token = token_data["access_token"]
                        st.session_state.authenticated = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Authentication failed: {str(e)}")
                        if hasattr(e, 'response'):
                            st.error(f"Response status: {e.response.status_code}")
                            st.error(f"Response body: {e.response.text}")
                        return
                else:
                    # Redirect to OpenShift OAuth
                    params = {
                        "client_id": CLIENT_ID,
                        "response_type": "code",
                        "redirect_uri": REDIRECT_URI,
                        "scope": "user:info user:check-access"
                    }
                    login_url = f"{AUTHORIZE_URL}?{urlencode(params)}"
                    st.markdown(f"""
                        <meta http-equiv="refresh" content="0;url={login_url}" />
                    """, unsafe_allow_html=True)
                    return

            return func(*args, **kwargs)
        return wrapper
    return decorator