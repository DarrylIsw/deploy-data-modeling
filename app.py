import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================
# Load TFLite model
# =====================================

interpreter = tf.lite.Interpreter(
    model_path="model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =====================================
# Preprocess Image
# =====================================

IMG_SIZE = 256

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image).astype(np.float32)

    # MobileNetV3 preprocessing
    img_array = (img_array / 127.5) - 1.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# =====================================
# Sigmoid
# =====================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# =====================================
# Predict
# =====================================

def predict(image):

    input_data = preprocess_image(image)

    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )

    interpreter.invoke()

    output_data = interpreter.get_tensor(
        output_details[0]['index']
    )

    logits = output_data[0][0]

    probability = sigmoid(logits)

    label = (
        "Malignant"
        if probability > 0.5
        else "Benign"
    )

    return label, probability

# =====================================
# Streamlit UI
# =====================================

st.set_page_config(
    page_title="Melanoma Classifier",
    layout="centered"
)

st.title("Melanoma Image Classifier")

st.write(
    "Upload a dermoscopic skin lesion image "
    "for melanoma classification."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Classifying..."):

        label, probability = predict(image)

    st.subheader(f"Prediction: {label}")

    st.write(
        f"Confidence: {probability * 100:.2f}%"
    )