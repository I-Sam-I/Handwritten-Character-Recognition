import os
import numpy as np


# Set the main directory
MAIN_DIRECTORY = 'data/by_class'

NO_EACH_TRAIN_CLASS = 300
NO_EACH_TEST_CLASS = 100
NO_EACH_VALIDATION_CLASS = 10


def main():
    # Get the classes and labels
    classes_hex, classes_labels = get_classes()

    # Get the training data
    training_data, training_labels = load_training_data(classes_hex, classes_labels)

    # Get the testing data
    testing_data, testing_labels = load_testing_data(classes_hex, classes_labels)

    # Convert lists to NumPy arrays
    # training_data = np.array(training_data)
    # training_labels = np.array(training_labels)
    # testing_data = np.array(testing_data)
    # testing_labels = np.array(testing_labels)

    # print(training_data.shape, training_labels.shape)
    # print(testing_data.shape, testing_labels.shape)

    # print(training_data[0], training_labels[0])
    # print(testing_data[0], testing_labels[0])


def get_classes(dir=MAIN_DIRECTORY):
    """
    Retrieves the classes hex and label values from the directory

    Args:
        dir (str, optional): The directory to get the classes from

    Raises:
        FileNotFoundError: If the directory does not exist

    Returns:
        Tuple: A tuple containing the hex and label values of the classes
    """
    
    class_hex = []
    class_labels = []
    if os.path.exists(dir):
        classes = os.listdir(dir)
        class_hex = sorted(classes)
        class_labels = sorted([int(c, 16) for c in class_hex])

    else:
        raise FileNotFoundError(f"Directory '{dir}' does not exist")
    
    return class_hex, class_labels


def load_training_data(classes, labels, dir=MAIN_DIRECTORY):
    """
    Retrieves the NO_EACH_TRAIN_CLASS training data from the directory

    Args:
        classes (List): A list of the classes to load the data from
        labels (List): A list of the labels to assign to the classes
        dir (str, optional): The directory to load the training data from

    Raises:
        FileNotFoundError: If the directory does not exist

    Returns:
        Tuple: A tuple containing the numpy arrays training data and labels
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
                folder = os.listdir(train_dir)
                np.random.shuffle(folder)
                for file in folder[:NO_EACH_TRAIN_CLASS]:
                    file_path = os.path.join(train_dir, file)
                    training_data.append(file_path)
                    training_labels.append(label)
            
            else:
                raise FileNotFoundError(
                    f"Directory '{train_dir}' does not exist")
    
    else:
        raise FileNotFoundError(f"Directory '{dir}' does not exist")

    return np.array(training_data), np.array(training_labels)


def load_testing_data(classes, labels, dir=MAIN_DIRECTORY):
    """
    Retrieves the NO_EACH_TEST_CLASS testing data from the directory

    Args:
        classes (List): A list of the classes to load the data from
        labels (List): A list of the labels to assign to the classes
        dir (str, optional): The directory to load the testing data from

    Raises:
        FileNotFoundError: If the directory does not exist

    Returns:
        Tuple: A tuple containing the numpy arrays testing data and labels
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
            
            # Loop through the testing data in the hsf directory
            if os.path.exists(test_dir):
                folder = os.listdir(test_dir)
                np.random.shuffle(folder)
                for file in folder[:NO_EACH_TEST_CLASS]:
                    file_path = os.path.join(test_dir, file)
                    testing_data.append(file_path)
                    testing_labels.append(label)
            
            else:
                raise FileNotFoundError(
                    f"Directory '{test_dir}' does not exist")
    
    else:
        raise FileNotFoundError(f"Directory '{dir}' does not exist")

    return np.array(testing_data), np.array(testing_labels)


def load_validation_data(classes, labels, dir=MAIN_DIRECTORY):
    """
    Retrieves the NO_EACH_VALIDATION_CLASS validation data from the directory

    Args:
        classes (List): A list of the classes to load the data from
        labels (List): A list of the labels to assign to the classes
        dir (str, optional): The directory to load the validation data from

    Raises:
        FileNotFoundError: If the directory does not exist

    Returns:
        Tuple: A tuple containing the numpy arrays validation data and labels
    """
    
    validation_data = []
    validation_labels = []
    
    if os.path.exists(dir):
        
        # Loop through the classes and labels
        for c, label in zip(classes, labels):
            class_dir = os.path.join(dir, c)
            
            # Choose a random hsf directory
            hsf_directories = get_hsf_directories(class_dir)
            hsf_dir = np.random.choice(hsf_directories)
            val_dir = os.path.join(class_dir, hsf_dir)
            
            # Loop through the validation data in the hsf directory
            if os.path.exists(val_dir):
                folder = os.listdir(val_dir)
                np.random.shuffle(folder)
                for file in folder[:NO_EACH_VALIDATION_CLASS]:
                    file_path = os.path.join(val_dir, file)
                    validation_data.append(file_path)
                    validation_labels.append(label)
            
            else:
                raise FileNotFoundError(
                    f"Directory '{val_dir}' does not exist")
    
    else:
        raise FileNotFoundError(f"Directory '{dir}' does not exist")

    return np.array(validation_data), np.array(validation_labels)


def get_hsf_directories(dir=MAIN_DIRECTORY):
    """
    Retrieves the hsf directories from the directory

    Args:
        dir (str, optional): The directory to get the hsf directories from

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
