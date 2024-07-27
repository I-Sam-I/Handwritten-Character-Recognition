from load_data import get_classes, load_training_data, load_testing_data, load_validation_data, load_all_data

import numpy as np
from tensorflow.keras.utils import to_categorical
import time
import cv2
from sklearn.preprocessing import LabelEncoder

LABEL_ENCODER = LabelEncoder()


def main():
    # start_time = time.time()

    class_hex, class_labels = get_classes()
    training_data, training_labels = load_training_data(class_hex, class_labels)
    testing_data, testing_labels = load_testing_data(class_hex, class_labels)
    
    # training_data = np.array(training_data)
    # training_labels = np.array(training_labels)
    # testing_data = np.array(testing_data)
    # testing_labels = np.array(testing_labels)
    
    # print(training_data.shape, training_labels.shape)
    # print(testing_data.shape, testing_labels.shape)
    
    # training_indices = np.random.choice(len(training_data), TRAINING_SIZE, replace=False)
    # testing_indices = np.random.choice(len(testing_data), TESTING_SIZE, replace=False)
    
    # print(training_data.shape, training_labels.shape)
    # print(testing_data.shape, testing_labels.shape)
    
    # training_data = training_data[training_indices]
    # training_labels = training_labels[training_indices]
    # testing_data = testing_data[testing_indices]
    # testing_labels = testing_labels[testing_indices]

    # image_preprocess_start = time.time()
    training_data = preprocess_images(training_data)
    testing_data = preprocess_images(testing_data)
    # image_preprocess_end = time.time()
    
    # label_preprocess_start = time.time()
    training_labels = preprocess_labels(training_labels)
    testing_labels = preprocess_labels(testing_labels)
    # label_preprocess_end = time.time()
    
    end_time = time.time()
    
    # print(training_data[334], training_labels[334])
    
    # print(f"Total time: {end_time - start_time:.2f} seconds")
    # print(f"Image preprocessing time: {image_preprocess_end - image_preprocess_start:.2f} seconds")
    # print(f"Label preprocessing time: {label_preprocess_end - label_preprocess_start:.2f} seconds")
    

def preprocess_data():
    """
    Preprocesses the data for training, testing, and validation.

    Returns:
        Tuple: A tuple containing the preprocessed training data and labels,
               preprocessed testing data and labels, and preprocessed validation
               data and labels.
    """

    # Load all the data
    class_hex, class_labels = get_classes()
    (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = load_all_data(class_hex, class_labels)

    # Preprocess the images
    training_data = preprocess_images(training_data)
    testing_data = preprocess_images(testing_data)
    validation_data = preprocess_images(validation_data)

    # Preprocess the labels
    training_labels = preprocess_labels(training_labels)
    testing_labels = preprocess_labels(testing_labels)
    validation_labels = preprocess_labels(validation_labels)

    return (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels)
    
    
def preprocess_images(data):
    """
    Preprocesses a list of image paths.

    Args:
        data (list): A list of image paths.

    Returns:
        numpy.ndarray: A numpy array containing the preprocessed images.
    """
    processed_data = []

    for img_path in data:
        # Load the image in grayscale mode
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize the image to 32x32
        image = cv2.resize(image, (32, 32))

        # Normalize the image data to the range [0, 1]
        img_array = image / 255.0

        # Add a new axis to make the shape (128, 128, 1)
        img_array = np.expand_dims(img_array, axis=-1)

        # Append the processed image to the list
        processed_data.append(img_array)
    
    return np.array(processed_data)


def preprocess_labels(labels):
    """
    Preprocesses the given labels by applying label encoding and one-hot encoding.

    Args:
        labels (array-like): The labels to be preprocessed.

    Returns:
        array-like: The preprocessed labels in one-hot encoded format.
    """

    labels = LABEL_ENCODER.fit_transform(labels)
    return to_categorical(labels)
    
    
if __name__ == '__main__':
    main()