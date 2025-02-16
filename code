import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import sys

# Load the pre-trained MobileNetV2 model from TensorFlow
model = MobileNetV2(weights='imagenet')

def load_and_preprocess_image(image_path):
    # Load image with PIL
    img = Image.open(image_path)
    
    # Resize the image to the target size expected by MobileNetV2 (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to a numpy array and add an extra batch dimension
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image to the format MobileNetV2 expects
    img_array = preprocess_input(img_array)
    
    return img_array

def predict_image_class(image_path):
    # Preprocess the image
    img_array = load_and_preprocess_image(image_path)
    
    # Predict the class of the image using MobileNetV2
    predictions = model.predict(img_array)
    
    # Decode the predictions to human-readable labels
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    # Print the top 3 predictions
    print("Top 3 Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i+1}. {label}: {score*100:.2f}%")
    
    return decoded_predictions

# Main function to run the app
if __name__ == '__main__':
    # Accept the image file path from the command line
    if len(sys.argv) < 2:
        print("Please provide an image path.")
        sys.exit(1)

    image_path = sys.argv[1]

    # Check if the file exists
    try:
        predict_image_class(image_path)
    except Exception as e:
        print(f"Error: {e}")
