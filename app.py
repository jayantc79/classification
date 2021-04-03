import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps


st.title("Image Classification with Teachable Machine Learning")
st.header("Normal X Ray Vs Pneumonia X Ray")
st.text("Upload a X Ray to detect it is normal or has pneumothorax")
# file upload and handling logic
uploaded_file = st.file_uploader("Choose a X Ray Image", type=['jpeg', 'png', 'jpg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    # image = Image.open(img_name).convert('RGB')
    st.image(image, caption='Uploaded a X Ray IMage.', use_column_width=True)
    st.write("")
    st.write("Classifying a X Ray Image - Normal Vs Pneumothorax.........hold tight")
    label = teachable_machine_classification(image, 'weights_file.h5')
    if label == 0:
        st.write("Hooray!! This X ray looks like doesn't have pneumothorax.")
    else:
        st.write("Oops!! This X ray looks like pneumothorax.")

