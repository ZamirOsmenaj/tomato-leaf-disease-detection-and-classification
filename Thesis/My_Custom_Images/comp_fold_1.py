import os
import numpy as np
import tensorflow as tf

def model_prediction(test_image, model): 
    # Load and resize the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))

    # Convert the image to a numpy array and normalize it
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0

    # Expand dimensions to fit the model input shape (batch size of 1)
    input_arr = np.expand_dims(input_arr, axis=0)
    
    # Make prediction
    prediction = model.predict(input_arr)
    
    # Get the maximum prediction value (confidence score)
    max_value = np.max(prediction)
    percentage = max_value * 100
    
    # Get the predicted class index
    result_index = np.argmax(prediction)
    
    return result_index, percentage

def compare_folders(folder1, folder2, threshold=95):
    folder1_predictions = []
    folder2_predictions = []
    differences = 0

    # Get list of images from both folders
    folder1_images = os.listdir(folder1)
    folder2_images = os.listdir(folder2)
    
    # Create a mapping of base names to their full paths for both folders
    folder1_image_map = {os.path.splitext(img_name)[0]: os.path.join(folder1, img_name) for img_name in folder1_images}
    folder2_image_map = {os.path.splitext(img_name)[0]: os.path.join(folder2, img_name) for img_name in folder2_images}

     # Load trained model
    model = tf.keras.models.load_model('../Models/80-10-10/CNN_trained_tomato_leaf_disease_model.keras')
   
    # Compare predictions based on matching base names (ignoring extensions)
    for base_name in folder1_image_map:
        if base_name in folder2_image_map:
            # Get the full paths for the matching images
            img_path_folder1 = folder1_image_map[base_name]
            img_path_folder2 = folder2_image_map[base_name]
            
            # Get predictions for both folders
            result_index1, confidence1 = model_prediction(img_path_folder1, model)
            result_index2, confidence2 = model_prediction(img_path_folder2, model)
            
            # Count if the predicted classes are different
            if result_index1 != result_index2:
                differences += 1
            
            # Store predictions
            folder1_predictions.append((base_name, result_index1, confidence1))
            folder2_predictions.append((base_name, result_index2, confidence2))
    
    return folder1_predictions, folder2_predictions, differences


# Folder paths
# folder1_path = "My_Tomato_Leaves (test)"
folder1_path = "My_Tomato_Leaves (test) - Cropped"
folder2_path = "My_Tomato_Leaves (test) - New Background"

# Call the function to compare the folders
folder1_predictions, folder2_predictions, differences = compare_folders(folder1_path, folder2_path)

# Display results
print(f"Number of images with different predictions: {differences}")
print("\n--- Predictions for Folder 1 (Cropped) ---")
for img_name, pred_class, confidence in folder1_predictions:
    print(f"Image: {img_name} | Predicted Class: {pred_class} | Confidence: {confidence:.2f}%")

print("\n--- Predictions for Folder 2 (Cropped and Background Removed) ---")
for img_name, pred_class, confidence in folder2_predictions:
    print(f"Image: {img_name} | Predicted Class: {pred_class} | Confidence: {confidence:.2f}%")
