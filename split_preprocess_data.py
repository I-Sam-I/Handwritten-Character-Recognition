from preprocess_data import preprocess_images, preprocess_labels
from load_data import get_classes, load_all_data, load_validation_data

import numpy as np


def main():
    # Load all the data
    class_hex, class_labels = get_classes()
    # (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = load_all_data(class_hex, class_labels)
    validation_data, validation_labels = load_validation_data(class_hex, class_labels)

    # digits_data, digits_labels = [], []
    # upper_data, upper_labels = [], []
    # lower_data, lower_labels = [], []

    # print(chr(validation_labels[0]))

    # for data, label in zip(validation_data, validation_labels):
    #     c = chr(label)

    #     if c.isdigit():
    #         digits_data.append(data)
    #         digits_labels.append(label)
    #     elif c.isupper():
    #         upper_data.append(data)
    #         upper_labels.append(label)
    #     elif c.islower():
    #         lower_data.append(data)
    #         lower_labels.append(label)
    #     else:
    #         raise ValueError(f"Invalid character '{c}'")
    
    # print(digits_data[:5], digits_labels[:5])
    # print(upper_data[:5], upper_labels[:5])
    # print(lower_data[:5], lower_labels[:5])


    # Split and preprocess the data and labels
    # training_data_dict = split_and_preprocess(training_data, training_labels)
    # testing_data_dict = split_and_preprocess(testing_data, testing_labels)
    
    validation_data_dict = split_and_preprocess(validation_data, validation_labels)

    print(validation_data_dict['digits']['data'][:5], validation_data_dict['digits']['labels'][:5])


def split_and_preprocess(data, labels):
    """
    Splits the data into digits, upper case letters, and lower case letters,
    and preprocesses the images and labels.

    Args:
        data (numpy.ndarray): The input data array.
        labels (numpy.ndarray): The corresponding labels array.

    Returns:
        dict: A dictionary containing the preprocessed data and labels for digits, 
              upper case letters, and lower case letters.
              The dictionary has the following structure:
              {
                  'digits': {'data': digits_data, 'labels': digits_labels},
                  'uppercase': {'data': uppercase_data, 'labels': uppercase_labels},
                  'lowercase': {'data': lowercase_data, 'labels': lowercase_labels}
              }
    """

    # Split the data
    (digits_data, digits_labels), (uppercase_data, uppercase_labels), (lowercase_data, lowercase_labels) = split_data(data, labels)

    # Preprocess the images
    digits_data = preprocess_images(digits_data)
    uppercase_data = preprocess_images(uppercase_data)
    lowercase_data = preprocess_images(lowercase_data)

    # Preprocess the labels
    digits_labels = preprocess_labels(digits_labels)
    uppercase_labels = preprocess_labels(uppercase_labels)
    lowercase_labels = preprocess_labels(lowercase_labels)

    return {
        'digits': {'data': digits_data, 'labels': digits_labels},
        'uppercase': {'data': uppercase_data, 'labels': uppercase_labels},
        'lowercase': {'data': lowercase_data, 'labels': lowercase_labels}
    }


def split_data(data, labels):
    """
    Splits the given data and labels into three separate groups based on the label type.

    Parameters:
    data (numpy.ndarray): The input data.
    labels (numpy.ndarray): The corresponding labels.

    Returns:
    tuple: A tuple containing three tuples, each representing a group of data and labels.
        The first tuple contains digits data and labels,
        the second tuple contains uppercase letters data and labels,
        and the third tuple contains lowercase letters data and labels.
    """
    # Create masks for digits, uppercase letters, and lowercase letters
    # [65, 53, 86] (ASCII values) -> ['A', '5', 'V'] (characters) -> [False, True, False] (digit check)
    digits_mask = np.char.isdigit(np.char.mod('%c', labels))
    upper_mask = np.char.isupper(np.char.mod('%c', labels))
    lower_mask = np.char.islower(np.char.mod('%c', labels))

    # Split the data based on the masks
    digits_data, digits_labels = data[digits_mask], labels[digits_mask]
    upper_data, upper_labels = data[upper_mask], labels[upper_mask]
    lower_data, lower_labels = data[lower_mask], labels[lower_mask]
    
    return (digits_data, digits_labels), (upper_data, upper_labels), (lower_data, lower_labels)


if __name__ == '__main__':
    main()