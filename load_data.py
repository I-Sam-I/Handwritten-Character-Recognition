import os
import numpy as np


def main():
    # Set the main directory
    MAIN_DIRECTORY = 'data/by_class'

    # Get the classes and labels
    classes_hex = sorted(get_classes(MAIN_DIRECTORY))
    classes_labels = sorted([chr(int(c, 16)) for c in classes_hex])

    # Get the training data
    training_data, training_labels = get_training_data(
        MAIN_DIRECTORY, classes_hex, classes_labels)

    # Get the testing data
    testing_data, testing_labels = get_testing_data(
        MAIN_DIRECTORY, classes_hex, classes_labels)

    # Convert lists to NumPy arrays
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    testing_data = np.array(testing_data)
    testing_labels = np.array(testing_labels)

    # print(training_data.shape, training_labels.shape)
    # print(testing_data.shape, testing_labels.shape)

    # print(training_data[0], training_labels[0])
    # print(testing_data[0], testing_labels[0])


def get_classes(dir):
    """Retrieves the classes from the directory

    Args:
        dir (str): The directory to get the classes from

    Raises:
        FileNotFoundError: If the directory does not exist

    Returns:
        List: A list of the classes in the directory
    """
    
    classes = []
    if os.path.exists(dir):
        classes = os.listdir(dir)
    else:
        raise FileNotFoundError(f"Directory '{dir}' does not exist")
    return classes


def get_training_data(dir, classes, labels):
    """Retrieves the training data from the directory

    Args:
        dir (str): The directory to get the training data from
        classes (List): A list of the classes to get the data from

    Raises:
        FileNotFoundError: If the directory does not exist

    Returns:
        Tuple: A tuple containing the training data and labels
    """

    training_data = []
    training_labels = []

    if os.path.exists(dir):
        # Loop through the classes and labels
        for c, label in zip(classes, labels):
            
            # Loop through the training data in each class
            train_dir = os.path.join(dir, c, f'train_{c}')
            if os.path.exists(train_dir):
                
                # Add the file path and label to the training data and labels
                for file in os.listdir(train_dir):
                    file_path = os.path.join(train_dir, file)
                    training_data.append(file_path)
                    training_labels.append(label)
            
            else:
                raise FileNotFoundError(
                    f"Directory '{train_dir}' does not exist")
    
    else:
        raise FileNotFoundError(f"Directory '{dir}' does not exist")

    return training_data, training_labels


def get_testing_data(dir, classes, labels):
    """Retrieves the testing data from the directory

    Args:
        dir (str): The directory to get the testing data from
        classes (List): A list of the classes to get the data from

    Raises:
        FileNotFoundError: If the directory does not exist

    Returns:
        Tuple: A tuple containing the testing data and labels
    """
    
    testing_data = []
    testing_labels = []
    
    if os.path.exists(dir):
        
        # Loop through the classes and labels
        for c, label in zip(classes, labels):
            class_dir = os.path.join(dir, c)
            
            # Choose a random hsf directory
            hsf_directories = get_hsf_directories(class_dir)
            hsf_dir = np.random.choice(hsf_directories)
            test_dir = os.path.join(class_dir, hsf_dir)
            
            # Loop through the testing data in each hsf directory
            if os.path.exists(test_dir):
                for file in os.listdir(test_dir):
                    file_path = os.path.join(test_dir, file)
                    testing_data.append(file_path)
                    testing_labels.append(label)
            
            else:
                raise FileNotFoundError(
                    f"Directory '{test_dir}' does not exist")
    
    else:
        raise FileNotFoundError(f"Directory '{dir}' does not exist")

    return testing_data, testing_labels


def get_hsf_directories(dir):
    """Retrieves the hsf directories from the directory

    Args:
        dir (str): The directory to get the hsf directories from

    Raises:
        FileNotFoundError: If the directory does not exist

    Returns:
        List: A list of the hsf directories in the directory
    """
    
    hsf = []
    
    if os.path.exists(dir):
        # Get all the directories in the directory
        dirs = os.listdir(dir)
        
        # Filter the directories to get the hsf directories
        hsf = [d for d in dirs if d.startswith('hsf') and os.path.isdir(os.path.join(dir, d))]
    
    else:
        raise FileNotFoundError(f"Directory '{dir}' does not exist")
    
    return hsf


if __name__ == '__main__':
    main()
