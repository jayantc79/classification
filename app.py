import streamlit as st
st.title("Image Classification with Teachable Machine Learning")
st.header("Normal X Ray Vs Pneumonia X Ray")
st.text("Upload a X Ray to detect it is normal or has pneumonia")
# file upload and handling logic
uploaded_file = st.file_uploader("Choose a X Ray Image", type="jpeg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
