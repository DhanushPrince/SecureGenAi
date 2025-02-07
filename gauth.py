import streamlit as st
import json
import os
import google.auth.transport.requests
import google.oauth2.id_token
from google_auth_oauthlib.flow import Flow
from urllib.parse import urlparse, parse_qs

# Load Google OAuth Client Secret JSON
CLIENT_SECRET = "client_secret.json"

st.title("Google Authentication in Streamlit")

# Function to create an OAuth flow
def get_auth_flow():
    return Flow.from_client_secrets_file(
        CLIENT_SECRET,
        scopes=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
        redirect_uri="http://localhost:8501"  # Update this for deployment
    )

# Function to handle login
def authenticate_google():
    flow = get_auth_flow()
    auth_url, _ = flow.authorization_url(prompt="consent")
    st.session_state["auth_flow"] = flow  # Save flow in session state
    st.write(f"[Login with Google]({auth_url})")

# Function to exchange the authorization code for tokens
def exchange_auth_code(auth_code):
    try:
        flow = st.session_state.get("auth_flow", get_auth_flow())
        flow.fetch_token(code=auth_code)
        credentials = flow.credentials

        # Verify ID Token
        id_info = google.oauth2.id_token.verify_oauth2_token(
            credentials.id_token, google.auth.transport.requests.Request(), flow.client_config["client_id"]
        )
        
        return id_info
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None

# Check if user is authenticated
if "user" not in st.session_state:
    st.session_state.user = None

# Capture query params from Google OAuth redirect
query_params = st.query_params
if "code" in query_params:
    auth_code = query_params["code"]
    user_info = exchange_auth_code(auth_code)
    if user_info:
        st.session_state.user = user_info

# Show login button if not logged in
if st.session_state.user is None:
    if st.button("Login with Google"):
        authenticate_google()

# Display user info if logged in
if st.session_state.user:
    st.success(f"Logged in as {st.session_state.user['email']}")
    st.image(st.session_state.user.get("picture", ""), width=100)
    st.write(f"Name: {st.session_state.user['name']}")

    # Example: Restrict content based on user email
    if st.session_state.user["email"] == "admin@example.com":
        st.write("Welcome, Admin! You have full access.")
    else:
        st.write("Welcome! You have limited access.")

    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()
