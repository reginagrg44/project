import streamlit as st
import tensorflow as tf
import numpy as np
import os  # Import os module for checking file existence

import h5py

# Open the .h5 file in read mode
f = h5py.File('my_model.h5', mode='r')

# Perform operations with the file
print(f.keys())  # Example to show the contents


# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(r"my_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    
    # Define the image path
    image_path = r"C:\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\greenlove.jpeg"
    
    # Check if the image exists before displaying it
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    else:
        st.error("Image file does not exist at the specified path!")

    st.markdown(""" ... """)  # Your existing markdown content

elif app_mode == "About":
    st.header("About")
    st.markdown(""" ... """)  # Your existing about content

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    
    if test_image and st.button("Predict"):
        # st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name =  ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success(f"Model is Predicting it's a {class_name[result_index]}")
