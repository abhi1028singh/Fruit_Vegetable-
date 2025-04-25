import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    
    # Load image as PIL Image
    image = Image.open(test_image)
    image = image.resize((64, 64))  # Resize the image to match model's expected input size
    
    input_arr = np.array(image)  # Convert the image to a numpy array
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert the image to a batch of one image
    
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    st.image(image_path)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chili pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        st.image(test_image, width=400, use_column_width=True)
    
        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction...")
            
            # Call the prediction function
            result_index = model_prediction(test_image)
            
            # Reading Labels
            with open("labels.txt") as f:
                content = f.readlines()
            
            label = [i.strip() for i in content]  # Remove newline characters
            st.success(f"Model is predicting it's a {label[result_index]}")
