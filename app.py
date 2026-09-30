import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.title("Sign Language MNIST Predictor")


# Direct reliable URL to Sign Language MNIST Test CSV
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/leekanghee/Sign-Language-MNIST/master/sign_mnist_test.csv"
    return pd.read_csv(url)


data = load_data()

st.sidebar.header("Options")
sample_idx = st.sidebar.slider("Select Image Index", 0, len(data) - 1, 0)

# Extract label and pixel values
label = data.iloc[sample_idx, 0]
pixels = data.iloc[sample_idx, 1:].values.reshape(28, 28)

# Map label index to alphabet letter (J=9 and Z=25 excluded in dataset)
letters = [chr(i) for i in range(65, 91) if i not in (74, 90)]
true_letter = letters[label] if label < len(letters) else str(label)

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"True Label: {true_letter} (Class {label})")
    fig, ax = plt.subplots()
    ax.imshow(pixels, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

with col2:
    st.subheader("Pixel Statistics")
    st.write(f"**Min Value:** {pixels.min()}")
    st.write(f"**Max Value:** {pixels.max()}")
    st.write(f"**Mean Intensity:** {pixels.mean():.2f}")
