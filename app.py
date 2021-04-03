import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps


st.title("Image Classification with Teachable Machine Learning")
st.header("Normal X Ray Vs Pneumonia X Ray")
st.text("Upload a X Ray to detect it is normal or has pneumonia")
# file upload and handling logic
uploaded_file = st.file_uploader("Choose a X Ray Image", type='png')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
#image = Image.open(img_name).convert('RGB')
    st.image(image, caption='Uploaded a X Ray IMage.', use_column_width=True)
    st.write("")
    st.write("Classifying a X Ray Image - Normal Vs Pneumonia.........hold tight")
    label = teachable_machine_classification(image, 'weights_file.h5')
    if label == 1:
        st.write("This X ray looks like having pneumonia.It has abnormal opacification.Needs further investigation by a Radiologist/Doctor.")
    else:
        st.write("Hooray!! This X ray looks normal.This X ray depicts clear lungs without any areas of abnormal opacification in the image")

