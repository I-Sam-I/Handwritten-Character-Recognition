from multi_model_classifier import create_multi_model_classifier, test_predict

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pickle import load


def main():
    # Load pre-trained CNN models
    models_dict = load_models()

    # Load the image
    image_path = 'data/words/uppercase/FIVE/FIVE/5.jpg'
    # image_path = 'data/words/lowercase/five/five/7.jpg'

    # save the images on each step

    # Preprocess the image
    image = preprocess_image(image_path)
    cv2.imwrite('images/preprocessed.png', image)

    # Extract letters from the contours
    letters = extract_letters(image)

    # Classify the letters
    classifier = create_multi_model_classifier(*list(models_dict.values())[:-1])
    predictions = classify_letters(letters, classifier)
    
    print('Letters:', predictions)


def load_models(models_directory='models/'):
    """
    Load the pre-trained models for digit recognition, uppercase letter recognition,
    lowercase letter recognition, and character classification.

    Args:
        models_directory (str, optional): The directory path where the models are stored. 
            Default is 'models/'.

    Returns:
        dict: A dictionary containing the loaded models.
            The keys are 'digit', 'uppercase', 'lowercase', and 'classifier'.
            The values are the corresponding loaded models.
    """
    # Load the models
    digit_model = load_model(f'{models_directory}/Digit_Model.keras')
    uppercase_model = load_model(f'{models_directory}/Uppercase_Model.keras')
    lowercase_model = load_model(f'{models_directory}/Lowercase_Model.keras')
    classifier_model = load_model(f'{models_directory}/Classifier_Model.keras')

    # Return as a dictionary
    return {
        'digit': digit_model,
        'uppercase': uppercase_model,
        'lowercase': lowercase_model,
        'classifier': classifier_model
    }


def preprocess_image(image_path):
    """
    Preprocesses an image for optical character recognition (OCR).

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The preprocessed image.

    """

    # Load the image
    image = cv2.imread(image_path)

    # Remove noise
    image = cv2.fastNlMeansDenoising(image,None,25,15,15)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 15)

    # Remove horizontal and vertical lines using Hough Line Transform
    clean = remove_lines_using_hough(thresh)

    return clean


def remove_lines_using_hough(binary_image):
    """
    Removes vertical and horizontal lines from a binary image using Hough Line Transform.

    Args:
        binary_image (numpy.ndarray): The binary image from which vertical and horizontal lines are to be removed.

    Returns:
        numpy.ndarray: The binary image with vertical and horizontal lines removed.
    """

    # Invert the binary image
    binary_image = ~binary_image

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(binary_image, 1, np.pi/180, threshold=75, minLineLength=79, maxLineGap=10)
    
    # Iterate through the detected lines
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Draw over the detected vertical and horizontal lines
                cv2.line(binary_image, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    # Invert the binary image back
    return ~binary_image


def find_letter_contours(binary_image):
    """
    Find contours of letters in a binary image.

    Args:
        binary_image (numpy.ndarray): The binary image containing letters.

    Returns:
        list: A list of contours representing the letters in the image.
    """

    # Find contours in the binary image
    contours, _ = cv2.findContours(~binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


#TODO: Properly extract the letters
def extract_letters(binary_image):
    """
    Extracts individual letters from a binary image.

    Args:
        binary_image (numpy.ndarray): The binary image containing handwritten text.

    Returns:
        list: A list of numpy arrays, each representing an individual letter.

    """
    
    # Find contours of the letters
    contours = find_letter_contours(binary_image)

    # Draw rectangles around the contours
    image_with_contours = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    # Extract the letters using the bounding rectangles
    letters = []
    for i, contour in enumerate(contours):

        # Get the coordinates of the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the letter image using the coordinates
        letter_image = binary_image[y:y+h, x:x+w]

        # Append the letter image to the list
        letters.append(letter_image)

        # Save the letter image
        cv2.imwrite(f'images/letter ({i}).png', letter_image)

        # Draw the bounding rectangle
        cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Save the image with contours
    cv2.imwrite('images/All Contours.png', image_with_contours)
    
    # Return the extracted letters
    return letters


def resize_and_normalize(letter_image, size=(32, 32)):
    """
    Resize and normalize a letter image.

    Args:
        letter_image (numpy.ndarray): The input letter image.
        size (tuple, optional): The desired size of the resized image. Default is (32, 32).

    Returns:
        numpy.ndarray: The resized and normalized image.

    """
    # Resize the letter image to the desired size
    resized_image = cv2.resize(letter_image, size)

    # Normalize the pixel values to [0, 1]
    normalized_image = resized_image / 255.0

    # Expand dimensions to match the input shape of the CNN
    normalized_image = np.expand_dims(normalized_image, axis=-1)
    
    return normalized_image


def classify_letters(letters, classifier):
    """
    Classify a list of handwritten letters using a given classifier.

    Args:
        letters (list): A list of handwritten letters.
        classifier: The classifier model used for prediction.

    Returns:
        list: A list of predicted characters corresponding to the input letters.
    """

    # Classify each letter
    predictions = []
    for letter in letters:

        # Preprocess the letter image
        processed_letter = resize_and_normalize(letter)

        # Add batch dimension   
        processed_letter = np.expand_dims(processed_letter, axis=0)

        # Predict the letter
        prediction = classifier(processed_letter)

        # Append the prediction to the list
        predictions.append(prediction)
    
    # Load the Label Encoder from the file
    with open('models/label_encoder.pkl', 'rb') as file:
        label_encoder = load(file)

    # Inverse transform the predictions to get the original ASCII labels
    predictions = label_encoder.inverse_transform(predictions)

    # Convert the ASCII labels to characters
    predictions = [chr(label) for label in predictions]

    return predictions


if __name__ == '__main__':
    main()