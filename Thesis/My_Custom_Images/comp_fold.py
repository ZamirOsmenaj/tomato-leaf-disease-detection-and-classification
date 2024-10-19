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
    confidence_rises = []
    confidence_decreases = []  # Add this to track decreases
    
    # Lists to store rise and decrease values for file output
    rise_values = []
    decrease_values = []

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
            
            # Calculate confidence rise or decrease
            if confidence2 > confidence1:
                rise_percentage = ((confidence2 - confidence1) / confidence1) * 100
                confidence_rises.append((base_name, confidence1, confidence2, rise_percentage))
                rise_values.append(f"{rise_percentage:.2f}")  # Add the rise value to the list
            elif confidence2 < confidence1:
                decrease_percentage = ((confidence1 - confidence2) / confidence1) * 100
                confidence_decreases.append((base_name, confidence1, confidence2, decrease_percentage))
                decrease_values.append(f"-{decrease_percentage:.2f}")  # Add the negative decrease value to the list
    
    # Write rises and decreases to a file
    with open("confidence_changes.txt", "w") as file:
        file.write("Rise: " + ", ".join(rise_values) + "\n")
        file.write("Decrease: " + ", ".join(decrease_values) + "\n")
    
    # Return the predictions, confidence rises, and confidence decreases
    return folder1_predictions, folder2_predictions, differences, confidence_rises, confidence_decreases


# Folder paths
# folder1_path = "My_Tomato_Leaves (test)"
folder1_path = "My_Tomato_Leaves (test) - Cropped"
folder2_path = "My_Tomato_Leaves (test) - New Background"

# Call the function to compare the folders
folder1_predictions, folder2_predictions, differences, confidence_rises, confidence_decreases = compare_folders(folder1_path, folder2_path)

# Display confidence rise information
if confidence_rises:
    print("\n--- Confidence Rises ---")
    total_rise = 0
    for img_name, conf1, conf2, rise_percent in confidence_rises:
        print(f"Image: {img_name} | Confidence1: {conf1:.2f}% | Confidence2: {conf2:.2f}% | Rise: {rise_percent:.2f}%")
        total_rise += rise_percent
    average_rise = total_rise / len(confidence_rises)
    print(f"\nNumber of images with confidence rise: {len(confidence_rises)}")
    print(f"Average rise in confidence: {average_rise:.2f}%")
else:
    print("\nNo confidence rises found.")

# Display confidence decrease information
if confidence_decreases:
    print("\n--- Confidence Decreases ---")
    total_decrease = 0
    for img_name, conf1, conf2, decrease_percent in confidence_decreases:
        print(f"Image: {img_name} | Confidence1: {conf1:.2f}% | Confidence2: {conf2:.2f}% | Decrease: {decrease_percent:.2f}%")
        total_decrease += decrease_percent
    average_decrease = total_decrease / len(confidence_decreases)
    print(f"\nNumber of images with confidence decrease: {len(confidence_decreases)}")
    print(f"Average decrease in confidence: {average_decrease:.2f}%")
else:
    print("\nNo confidence decreases found.")
