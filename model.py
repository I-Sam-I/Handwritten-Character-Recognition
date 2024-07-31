from preprocess_data import preprocess_data

import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    # Load the data
    (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = preprocess_data()
    
    print(training_data.shape, training_labels.shape)
    print(testing_data.shape, testing_labels.shape)
    print(validation_data.shape, validation_labels.shape)
    print('\n\n\n')
    
    # print(training_data[0])
    
    # for i in training_data:
    #     if i.shape != (64, 64, 1):
    #         print(i.shape)
    
    # for i in range(5):
    #     plt.imshow(training_data[i], cmap='gray')
    #     plt.title(training_labels[i])
    #     plt.show()
    
    # Create the model
    model = create_model()
    print(model.summary())

    # Train the model
    history = model.fit(training_data, training_labels, epochs=20, validation_data=(validation_data, validation_labels))
    
    # Evaluate the model
    print('\n\n')
    loss, accuracy = model.evaluate(testing_data, testing_labels)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}\n')

    # Plot the model
    plot_model(history)

    # Load in previous model
    past_model = tf.keras.models.load_model('model.keras')

    # Compare the two models
    past_loss, past_accuracy = past_model.evaluate(testing_data, testing_labels)
    print(f'Past Test Loss: {past_loss}, Past Test Accuracy: {past_accuracy}')
    print(past_model.summary())

    # Save the model
    if input('\n\nDo you want to save the model? (y/n) ').strip().lower() == 'y':
        model.save('models/model.keras')
     

def plot_model(history, filename='model_accuracy_loss.png'):
    """
    Saves the plot of the training and validation accuracy as well as the training and validation loss of a model.

    Parameters:
    history (keras.callbacks.History): The history object returned by the `fit` method of a Keras model.
    filename (str, optional): The name of the file to save the plot

    Returns:
    None
    """

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # save the plot
    plt.savefig(f'plots/{filename}')


def create_model():    
    """
    Creates a convolutional neural network model for handwritten OCR.

    Returns:
        model (tf.keras.models.Sequential): The created and compiled model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(32, 32, 1)),
        
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        # tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        # tf.keras.layers.MaxPooling2D((2, 2)),

        # tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),        
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(62, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


if __name__ == '__main__':
    main()

