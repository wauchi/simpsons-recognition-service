import streamlit as st
from fastai.vision import *

cwd = os.getcwd()
model = load_learner(cwd + '/model')


def predict(x):
    category, tensors, proba = model.predict(x)
    return category, tensors, proba


st.title('Simpsons-Classifier')

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = open_image(img_file_buffer)
    cat, tensors, proba = predict(image)

    st.image(img_file_buffer, caption="Processed image", use_column_width=True)
    st.write(str(cat).replace("_", " ").title())
