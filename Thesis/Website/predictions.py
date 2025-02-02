import streamlit as st
from streamlit_option_menu import option_menu

import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Tensorflow Model Prediction
def model_prediction(model, test_image):    
    # Load and resize the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))

    # Convert the image to a numpy array and normalize it
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0

    # Convert single image to a batch
    input_arr = np.expand_dims(input_arr, axis=0)
    
    # Make prediction
    prediction = model.predict(input_arr)

    # Get the maximum value
    max_value = np.max(prediction)

    # Convert to percentage
    percentage = max_value * 100

    result_index = np.argmax(prediction)

    return result_index, percentage

def predict_page(model):
    st.title("Tomato Leaf Disease Detection")
    test_image = st.file_uploader("Choose file:")

    # Dictionary mapping diseases to enriched treatments
    treatments = {
        "Tomato___Bacterial_spot": (
            "Bacterial spot is caused by Xanthomonas bacteria. "
            "To manage this disease, use copper-based bactericides regularly and avoid overhead watering "
            "to reduce leaf wetness. Practice crop rotation and remove and destroy infected plant debris."
        ),
        "Tomato___Early_blight": (
            "Early blight is caused by Alternaria solani. Remove infected leaves immediately to prevent spread. "
            "Apply fungicides such as chlorothalonil or mancozeb. Improve air circulation and avoid overhead watering."
        ),
        "Tomato___Late_blight": (
            "Late blight is caused by Phytophthora infestans. Regularly apply fungicides like chlorothalonil or copper-based sprays. "
            "Remove and destroy infected plants. Avoid wet foliage by watering at the base and ensure good air circulation."
        ),
        "Tomato___Leaf_Mold": (
            "Leaf mold is caused by the fungus Passalora fulva. Ensure good air circulation by spacing plants properly. "
            "Remove affected leaves and use fungicides such as chlorothalonil. Avoid wet leaves by watering at the base."
        ),
        "Tomato___Septoria_leaf_spot": (
            "Septoria leaf spot is caused by Septoria lycopersici. Remove and destroy infected leaves. "
            "Apply fungicides like chlorothalonil or copper-based sprays. Practice crop rotation and avoid overhead watering."
        ),
        "Tomato___Spider_mites Two-spotted_spider_mite": (
            "Spider mites are tiny pests that cause stippling on leaves. Use miticides like abamectin or neem oil. "
            "Encourage natural predators such as ladybugs and predatory mites. Ensure proper watering and avoid plant stress."
        ),
        "Tomato___Target_Spot": (
            "Target spot is caused by the fungus Corynespora cassiicola. Apply fungicides like chlorothalonil or copper-based sprays. "
            "Remove and destroy infected plant debris. Practice crop rotation and ensure good air circulation."
        ),
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
            "Tomato yellow leaf curl virus is transmitted by whiteflies. Use resistant varieties and control whitefly populations "
            "with insecticides like imidacloprid or insecticidal soaps. Use reflective mulches to repel whiteflies and remove infected plants."
        ),
        "Tomato___Tomato_mosaic_virus": (
            "Tomato mosaic virus is a highly contagious virus. Remove and destroy infected plants. "
            "Sanitize tools and equipment regularly. Avoid handling plants when they are wet and use virus-free seeds."
        ),
        "Tomato___healthy": "No treatment needed, the plant is healthy."
    }

    if st.button("Predict"):
        if test_image is not None:
             # Get the file extension
            file_extension = os.path.splitext(test_image.name)[1]

            # Check if the file extension is acceptable
            if file_extension.lower() in ['.png', '.jpg', '.jpeg']:
                st.markdown("**Model Prediction:**")

                result_index, percentage = model_prediction(model, test_image)

                test_data_dir = '../Tomato_Leaves_Dataset/80-10-10/test'
                diseases = os.listdir(test_data_dir)
                diseases.sort()

                disease_name = diseases[result_index]
                treatment = treatments.get(disease_name, "No treatment recommendation available.")
                
                if disease_name == "Tomato___healthy":
                    if percentage > 70:
                        st.success(disease_name + str(f" ---> {percentage:.2f}%"))
                    else:
                        st.warning(disease_name + str(f" ---> {percentage:.2f}%"))
                else:
                    if percentage > 70:
                        st.error(disease_name + str(f" ---> {percentage:.2f}%")) 
                    else:
                        st.warning(disease_name + str(f" ---> {percentage:.2f}%"))
                
                # Display the uploaded image in a larger size (400x300)
                image = Image.open(test_image)
                st.image(image.resize((400, 300)), caption='Uploaded Image', use_container_width=True)

                st.markdown("### Treatment Recommendation")
                st.write(treatment)
            else:
                st.error("Unsupported file format. Please upload a PNG, JPG, or JPEG file.")
        else:
            st.error("Please upload an image..")

def home_page():
    st.markdown("""
        # Tomato Leaf Disease Identifier

        ## *Welcome to Tomato Leaf Disease Identifier!*

        Our cutting-edge AI-driven platform specializes in the rapid and precise identification of diseases affecting tomato leaves. By simply uploading an image of your tomato plant’s leaves, our sophisticated algorithms can diagnose potential diseases within seconds. This tool is specifically tailored for tomato plants, ensuring highly accurate results. Whether you're a home gardener, a farmer, or involved in agricultural research, our platform is designed to help you maintain the health of your tomato crops efficiently and effectively.

        **How It Works:**
        - **Upload Your Image:** Select an image of the affected tomato leaves from your device.
        - **AI Analysis:** Our advanced AI model processes the image to detect any signs of disease.
        - **Instant Diagnosis:** Receive an instant diagnosis along with detailed information about the detected disease.
        - **Expert Recommendations:** Get personalized treatment recommendations to help you manage and cure the disease.

        By leveraging the power of AI, we aim to support sustainable agriculture and help you achieve better yields with healthier plants.
    """)

    st.image("media/home_page.jpeg")

def features():
    st.markdown("""
        # Key Features of Our Platform

        - **Easy Image Upload:** Our platform provides seamless image upload options, allowing you to easily choose or drag and drop images of tomato leaves. This ensures a hassle-free experience for users.
        - **Accurate Disease Detection:** Utilizing state-of-the-art algorithms, our system can accurately identify a variety of diseases specific to tomato plants. Our model is trained on extensive datasets to ensure high precision.
        - **Treatment Recommendations:** Once a disease is detected, our platform offers personalized treatment recommendations tailored to the specific disease. This helps you take immediate and effective action.
        - **User-Friendly Interface:** The platform is designed with simplicity in mind, offering an intuitive and user-friendly interface. Whether you're tech-savvy or a novice, you’ll find our platform easy to navigate and use.

        Our features are designed to provide a comprehensive solution for tomato disease management, ensuring that you can protect your plants with minimal effort.
    """)

    st.image("media/features.jpeg")

def recommendations():
    st.markdown("""
        # Personalized Treatment Recommendations

        Our platform goes beyond disease identification by offering customized treatment recommendations to address the specific needs of your tomato plants. Based on the diagnosis, you will receive:

        - **Detailed Information:** Comprehensive details about the identified disease, including symptoms, causes, and prevention measures.
        - **Treatment Options:** Tailored treatment plans and remedies to effectively manage and cure the disease.
        - **Preventive Tips:** Tips and best practices to prevent the recurrence of the disease and ensure the long-term health of your tomato plants.
        - **Resource Links:** Links to additional resources and expert articles for further reading and understanding.

        By providing actionable insights and expert advice, we aim to help you maintain healthy tomato plants and achieve optimal growth and productivity.
    """)

    st.image("media/recommendations.jpeg")

def about_us():
    st.markdown("""
        # About Us
        
        We are a dedicated team of plant enthusiasts and AI experts committed to promoting plant health and sustainability. Our journey began with a shared passion for agriculture and technology, leading us to develop this innovative platform that addresses the challenges faced by tomato growers worldwide.

        **Our Mission:**
            To empower farmers, gardeners, and agricultural professionals with cutting-edge tools for accurate disease identification and effective management, ultimately contributing to healthier crops and increased productivity.

        **Our Vision:**
            To revolutionize plant disease diagnosis through advanced AI technology, making it accessible and easy to use for everyone involved in tomato cultivation.

        **Our Team:**
            Comprised of agronomists, data scientists, and software engineers, our team combines expertise from various fields to deliver a robust and reliable platform. We believe in continuous improvement and regularly update our models and features based on the latest research and user feedback.

    """)

    st.image("media/about_us.jpeg")

def main():
    with st.sidebar: # If this is removed, this it goes straight to the page.
        selected = option_menu(
            menu_title = "Main Menu",
            options = ["Home", "Features", "Recommendations", "Disease Recognision", "About Us"],
            icons = ["house-fill", "list-stars", "bookmark-fill", "eyedropper", "people-fill"],
            menu_icon = "cast",
            default_index = 0,
            orientation = "vertical",
        )

    if selected == "Home":
        home_page()
    elif selected == "Features":
        features()
    elif selected == "Recommendations":
        recommendations()
    elif selected == "Disease Recognision":
        # Load trained model for making predictions
        model = tf.keras.models.load_model('../Models/80-10-10/CNN_trained_tomato_leaf_disease_model.keras')
    
        predict_page(model)
    else:
        about_us()

if __name__ == "__main__":
    main()
