import streamlit as st
from PIL import Image
from mnist_classify import worker

st.title('Digits Classification (MNIST based)')

image_file = st.file_uploader('Load an image', type=['png', 'jpg'])

if image_file is not None:
    image = Image.open(image_file)
    model_input, digit, certainty, vector = worker(image)
    st.markdown("Model input (x4)")
    st.image(Image.fromarray(model_input.squeeze().astype("uint8")).resize((model_input.shape[0]*4,model_input.shape[0]*4),Image.Resampling.NEAREST))
    st.markdown(f"Determined as digit **`{digit}`** with certainty **`{certainty:.3f}`**")
    st.markdown(f"Full vector: ")
    msg = []
    for i in range(len(vector)):
        msg.append(f"{i}:{vector[i]:.2f}")
    msg = " ".join(msg)
    st.text(msg)
