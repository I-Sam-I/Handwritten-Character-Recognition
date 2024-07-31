from split_preprocess_data import split_and_preprocess
from load_data import get_classes, load_all_data, load_training_data
from preprocess_data import preprocess_images, preprocess_labels
from model import plot_model
from pickle import load
from scipy.special import softmax

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split


def main():
    # Load all the data
    class_hex, class_labels = get_classes()
    (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = load_all_data(class_hex, class_labels)
    print('.')
    classifier_dict = load_classifier_data(training_data, training_labels)
    print('!')

    # Split and Preprocess the data
    training_dict = split_and_preprocess(training_data, training_labels)
    testing_dict = split_and_preprocess(testing_data, testing_labels)
    validation_dict = split_and_preprocess(validation_data, validation_labels)
    print('?')

    # Create the models
    digit_model, uppercase_model, lowercase_model, classifier_model = create_models()

    # Train the models
    train_model(digit_model, training_dict['digits'], testing_dict['digits'], validation_dict['digits'], epochs=50)
    train_model(uppercase_model, training_dict['uppercase'], testing_dict['uppercase'], validation_dict['uppercase'], epochs=100)
    train_model(lowercase_model, training_dict['lowercase'], testing_dict['lowercase'], validation_dict['lowercase'], epochs=100)
    train_model(classifier_model, classifier_dict['training'], classifier_dict['testing'], classifier_dict['validation'], epochs=200)


def train_model(model, training_dict, testing_dict, validation_dict, epochs=20, plot=True):
    """
    Trains a given model using the provided training, testing, and validation data.

    Args:
        model (keras.Model): The model to be trained.
        training_dict (dict): A dictionary containing the training data and labels.
        testing_dict (dict): A dictionary containing the testing data and labels.
        validation_dict (dict): A dictionary containing the validation data and labels.
        epochs (int, optional): The number of epochs to train the model (default is 20).
        plot (bool, optional): Whether to plot the model's accuracy and loss (default is True).

    Returns:
        None
    """

    # Print Model Summary
    model_name = model.name
    print(model.summary())

    # Train Model
    history = model.fit(
        training_dict['data'], training_dict['labels'], 
        epochs=epochs,
        validation_data=(validation_dict['data'], validation_dict['labels']))
    
    # Plot Model
    if plot:
        plot_model(history, f"{model_name}'s Accuracy and Loss")
    
    # Evaluate Model
    print('\n\n\n')
    loss, accuracy = model.evaluate(testing_dict['data'], testing_dict['labels'])
    print(f'{model_name}: Loss: {loss:.3f},\tAccuracy: {accuracy:.3f}')

    # Save Model
    if input(f"\nSave Model ({model_name})? (y/n): ").strip().lower().startswith('y'):
        model.save(f"models/{model_name}.keras")


def create_models():
    """
    Creates and returns multiple models for digit recognition, uppercase letter recognition, lowercase letter recognition, and classification.

    Returns:
        tuple: A tuple containing the following models:
            - digit_model: A model for digit recognition.
            - uppercase_model: A model for uppercase letter recognition.
            - lowercase_model: A model for lowercase letter recognition.
            - classifier_model: A model for classification.
    """

    # Create the models with the specified number of output classes
    digit_model = create_basic_cnn(10)          # 10 digits (0-9)
    uppercase_model = create_basic_cnn(26)      # 26 uppercase letters (A-Z)
    lowercase_model = create_basic_cnn(26)      # 26 lowercase letters (a-z)
    classifier_model = create_advanced_cnn(3)   # 3 classes (digits, uppercase, lowercase)

    # Set the model names
    digit_model.name = 'Digit_Model'
    uppercase_model.name = 'Uppercase_Model'
    lowercase_model.name = 'Lowercase_Model'
    classifier_model.name = 'Classifier_Model'

    # Return the models
    return digit_model, uppercase_model, lowercase_model, classifier_model


def create_basic_cnn(output_classes):
    """
    Create a basic convolutional neural network (CNN) model.

    Args:
        output_classes (int): The number of output classes.

    Returns:
        keras.models.Sequential: The compiled CNN model.

    """

    model = Sequential([
        Input(shape=(32, 32, 1)),

        Conv2D(16, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(64, activation='relu'),
        Dropout(0.25),

        Dense(64, activation='relu'),
        Dropout(0.25),

        Dense(output_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def create_advanced_cnn(output_classes):
    """
    Creates an advanced convolutional neural network (CNN) model for image classification.

    Args:
        output_classes (int): The number of output classes for the classification task.

    Returns:
        model (Sequential): The compiled CNN model.

    """

    model = Sequential([
        Input(shape=(32, 32, 1)),

        # Conv2D(16, (2, 2), activation='relu'),
        # BatchNormalization(),
        # Conv2D(16, (2, 2), activation='relu'),
        # BatchNormalization(),
        # MaxPooling2D((2, 2)),
        # Dropout(0.25),

        Conv2D(32, (5, 5), activation='relu'),
        BatchNormalization(),
        Conv2D(32, (4, 4), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (2, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(output_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def create_multi_model_classifier(digit_model, uppercase_model, lowercase_model, classifier_model):
    """
    Creates a multi-model classifier function that classifies and predicts handwritten characters.

    Args:
        digit_model (model): The model for classifying digits.
        uppercase_model (model): The model for classifying uppercase letters.
        lowercase_model (model): The model for classifying lowercase letters.
        classifier_model (model): The model for classifying the type of character.

    Returns:
        function: A function that takes an image as input and returns the predicted character.

    """

    def classify_and_predict(image):        
        # Classify the image
        class_prediction = classifier_model.predict(image)
        class_index = np.argmax(class_prediction)
        
        # Route to the appropriate model
        if class_index == 0:
            prediction = digit_model.predict(image)
        elif class_index == 1:
            prediction = uppercase_model.predict(image)
        else:
            prediction = lowercase_model.predict(image)
        
        return np.argmax(prediction)
    
    return classify_and_predict


def load_classifier_data(training_data, training_labels):
    """
    Load and preprocess the classifier data.

    Returns:
        dict: A dictionary containing the preprocessed training, testing, and validation data and labels.
            The dictionary has the following structure:
            {
                'training': {'data': training_data, 'labels': training_labels},
                'testing': {'data': testing_data, 'labels': testing_labels},
                'validation': {'data': validation_data, 'labels': validation_labels}
            }
    """
    
    # Declare lists to store the data and labels
    data = []
    labels = []

    # Use training data to create the classifier data
    for d, label in zip(training_data, training_labels):
        c = chr(label)  # Convert the label to a character
        
        data.append(d)  # Ensures the data and labels are in the same order

        if c.isdigit(): labels.append(0)
        elif c.isupper(): labels.append(1)
        elif c.islower(): labels.append(2)
        else: raise ValueError(f"Invalid character '{c}'")
    
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split the data into training and testing sets
    training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, test_size=0.2)

    # Further split the testing data into validation and testing sets
    testing_data, validation_data, testing_labels, validation_labels = train_test_split(testing_data, testing_labels, test_size=0.1)

    # Preprocess data
    training_data = preprocess_images(training_data)
    testing_data = preprocess_images(testing_data)
    validation_data = preprocess_images(validation_data)

    # Preprocess labels
    training_labels = preprocess_labels(training_labels)
    testing_labels = preprocess_labels(testing_labels)
    validation_labels = preprocess_labels(validation_labels)

    # Return the data and labels as nested dictionaries
    return {
        'training': {'data': training_data, 'labels': training_labels},
        'testing': {'data': testing_data, 'labels': testing_labels},
        'validation': {'data': validation_data, 'labels': validation_labels}
    }


#! This function is a testing function
def test_predict(digit_model, uppercase_model, lowercase_model):
    def predict(image):
        """
        Predicts the character in the given image using the provided models.

        Args:
            image (numpy.ndarray): The image to be classified.

        Returns:
            str: The predicted character.
        """

        # Predict the character
        digit_prediction = digit_model.predict(image)
        uppercase_prediction = uppercase_model.predict(image)
        lowercase_prediction = lowercase_model.predict(image)

        # Convert to probabilities
        # digit_probabilities = softmax(digit_prediction, axis=1)
        # uppercase_probabilities = softmax(uppercase_prediction, axis=1)
        # lowercase_probabilities = softmax(lowercase_prediction, axis=1)

        # Get the predictions
        digit_class = np.argmax(digit_prediction)
        uppercase_class = np.argmax(uppercase_prediction)
        lowercase_class = np.argmax(lowercase_prediction)

        # Get the probabilities
        # digit_confidence = digit_probabilities[0][digit_class]
        # uppercase_confidence = uppercase_probabilities[0][uppercase_class]
        # lowercase_confidence = lowercase_probabilities[0][lowercase_class]

        digit_confidence = digit_prediction[0][digit_class]
        uppercase_confidence = uppercase_prediction[0][uppercase_class]
        lowercase_confidence = lowercase_prediction[0][lowercase_class]

        # Make it into a list
        predictions = [
            (digit_class, digit_confidence),
            (uppercase_class, uppercase_confidence),
            (lowercase_class, lowercase_confidence)
        ]

        print(predictions)
        best_prediction = max(predictions, key=lambda x: x[1])
        print(best_prediction) #! Debugging

        # Load the label encoder
        label_encoder = load(open('models/label_encoder.pkl', 'rb'))

        # Get the best class as a character
        best_class = best_prediction[0]
        best_label = label_encoder.inverse_transform([best_class])[0]
        best_character = chr(best_label)

        return best_character

    return predict


if __name__ == '__main__':
    main()