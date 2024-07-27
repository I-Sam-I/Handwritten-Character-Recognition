from split_preprocess_data import split_and_preprocess
from load_data import get_classes, load_all_data, load_testing_data
from preprocess_data import preprocess_images, preprocess_labels
from model import plot_model

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split

def main():
    # Load all the data
    # class_hex, class_labels = get_classes()
    # (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = load_all_data(class_hex, class_labels)
    classifier_dict = load_classifier_data()

    # Split and Preprocess the data
    # training_dict = split_and_preprocess(training_data, training_labels)
    # testing_dict = split_and_preprocess(testing_data, testing_labels)
    # validation_dict = split_and_preprocess(validation_data, validation_labels)

    # Create the models
    digit_model, uppercase_model, lowercase_model, classifier_model = create_models()

    # Train the models
    # train_model(digit_model, training_dict['digits'], testing_dict['digits'], validation_dict['digits'])
    # train_model(uppercase_model, training_dict['uppercase'], testing_dict['uppercase'], validation_dict['uppercase'], epochs=50)
    # train_model(lowercase_model, training_dict['lowercase'], testing_dict['lowercase'], validation_dict['lowercase'], epochs=50)
    train_model(classifier_model, classifier_dict['training'], classifier_dict['testing'], classifier_dict['validation'], epochs=200)


def train_model(model, training_dict, testing_dict, validation_dict, epochs=20, plot=True):
    print(model.summary())

    # Train Model
    history = model.fit(
        training_dict['data'], training_dict['labels'], 
        epochs=epochs,
        validation_data=(validation_dict['data'], validation_dict['labels']),
        batch_size=32)
    
    # Plot Model
    if plot:
        plot_model(history, f"{model.name}'s Accuracy and Loss")
    
    # Evaluate Model
    print('\n\n\n')
    loss, accuracy = model.evaluate(testing_dict['data'], testing_dict['labels'])
    print(f'{model.name}: Loss: {loss:.3f},\tAccuracy: {accuracy:.3f}')

    # Save Model
    if input(f"Save Model ({model.name})? (y/n): ").strip().lower() == 'y':
        model.save(f"models/{model.name}.keras")


def create_models():
    digit_model = create_basic_cnn(10)
    uppercase_model = create_basic_cnn(26)
    lowercase_model = create_basic_cnn(26)
    classifier_model = create_advanced_cnn(3)

    digit_model.name = 'Digit_Model'
    uppercase_model.name = 'Uppercase_Model'
    lowercase_model.name = 'Lowercase_Model'
    classifier_model.name = 'Classifier_Model'

    return digit_model, uppercase_model, lowercase_model, classifier_model


def create_basic_cnn(output_classes):
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
    def classify_and_predict(image):
        # Preprocess the image
        image = preprocess_images(np.array([image]))
        
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


def load_classifier_data():
    # Load a data set
    class_hex, class_labels = get_classes()
    data, labels = load_testing_data(class_hex, class_labels)
    updated_labels = []

    for label in labels:
        c = chr(label)
        
        if c.isdigit():
            updated_labels.append(0)

        elif c.isupper():
            updated_labels.append(1)

        elif c.islower():
            updated_labels.append(2)

        else:
            raise ValueError(f"Invalid character '{c}'")
    
    updated_labels = np.array(updated_labels)

    training_data, testing_data, training_labels, testing_labels = train_test_split(data, updated_labels, test_size=0.2)
    validation_data, testing_data, validation_labels, testing_labels = train_test_split(testing_data, testing_labels, test_size=0.9)

    # Preprocess data and labels
    training_data = preprocess_images(training_data)
    testing_data = preprocess_images(testing_data)
    validation_data = preprocess_images(validation_data)

    training_labels = preprocess_labels(training_labels)
    testing_labels = preprocess_labels(testing_labels)
    validation_labels = preprocess_labels(validation_labels)

    return {
        'training': {'data': training_data, 'labels': training_labels},
        'testing': {'data': testing_data, 'labels': testing_labels},
        'validation': {'data': validation_data, 'labels': validation_labels}
    }


if __name__ == '__main__':
    main()