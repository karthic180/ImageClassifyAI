it is a Python-based application that uses a pre-trained AI model (MobileNetV2) to classify images. The app takes an image file as input and predicts the top 3 possible classes for the image with confidence scores.

## Features

- Classify images using a pre-trained deep learning model (MobileNetV2).
- Returns the top 3 predicted labels along with their confidence scores.
- Simple to use via the command line interface.

## Prerequisites

Before running the application, ensure that you have the following installed:

- Python 3.x
- TensorFlow
- Pillow (for image handling)
- NumPy

You can install the required libraries using pip:

```
pip install tensorflow pillow numpy
```

## How to Use

### 1. Clone the Repository or Download the Code
Clone the repository or download the script to your local machine.


### 2. Run the Application
After installing the dependencies, you can run the app from the command line by providing an image file as an argument.

```
python image_recognition.py path_to_your_image.jpg
```

### 3. Output
The app will display the top 3 predicted classes along with their confidence scores.

**Example Output:**

```
Top 3 Predictions:
1. cat: 98.23%
2. tabby: 1.54%
3. tiger_cat: 0.23%
```

### 4. File Path
Make sure to provide the correct path to the image you want to classify.

## How it Works

- **Model**: The app uses the `MobileNetV2` model, which is a lightweight deep learning model for image classification.
- **Preprocessing**: The input image is resized to 224x224 pixels and preprocessed to fit the model's requirements.
- **Prediction**: The model then classifies the image and returns the top 3 predicted classes.

## License

This project is licensed under the MIT License
