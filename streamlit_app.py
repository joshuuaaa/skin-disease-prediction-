import streamlit as st
import os
import shutil
from PIL import Image
import tempfile
import sys

# Ensure we can import from backend and models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend import database, auth
from models import model_utils

# --- Config ---
st.set_page_config(page_title="Skin Disease Prediction", layout="centered")

# --- Session State ---
if 'user' not in st.session_state:
    st.session_state.user = None

# --- database ---
# Create tables if not exist (though main.py did this, good to ensure)
database.Base.metadata.create_all(bind=database.engine)

def get_db():
    return database.SessionLocal()

# --- Auth Functions ---
def login_user(email, password):
    db = get_db()
    try:
        user = auth.get_user(db, email=email)
        if user and auth.verify_password(password, user.hashed_password):
            return user
        return None
    finally:
        db.close()

def signup_user(email, password):
    db = get_db()
    try:
        # Check if user exists
        if auth.get_user(db, email=email):
            return False, "Email already registered"
        
        user_create = auth.UserCreate(email=email, password=password)
        auth.create_user(db, user_create)
        return True, "Account created successfully"
    except Exception as e:
        return False, str(e)
    finally:
        db.close()

# --- UI ---

st.title("Skin Disease Prediction AI")

if not st.session_state.user:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.header("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.session_state.user = user
                st.success(f"Welcome back, {user.email}!")
                st.rerun()
            else:
                st.error("Invalid email or password")

    with tab2:
        st.header("Sign Up")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Create Account"):
            success, msg = signup_user(new_email, new_password)
            if success:
                st.success(msg)
            else:
                st.error(msg)

else:
    # Logged In View
    st.sidebar.write(f"Logged in as: **{st.session_state.user.email}**")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    st.markdown("### Upload a skin image for analysis")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                # Save temp file for model
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    label, confidence = model_utils.predict_single_image(tmp_path)
                    
                    st.markdown(f"## Prediction: **{label}**")
                    st.progress(confidence)
                    st.write(f"Confidence: {confidence:.1%}")
                    
                    st.warning("⚠️ This is an AI prediction and NOT a medical diagnosis. Please consult a dermatologist.")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
