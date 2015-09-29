def load_data(path='/home/cl/Downloads/data_kaggle_dr/train',
              filename_labels='/home/cl/Downloads/data_kaggle_dr/trainLabels.csv'):
    """Load all training data together with labels of Kaggle's Diabetic
    Retinopathy competition and split it into training and test sets

    Input
    =====

    :type path: string
    :param path: directory with unzipped images

    :type filename_labels: string
    :param filename_labels: csv file with labels


    Output - memory mapped arrays
    ======

    :rtype X_train: array of shape (examples, channels, rows, columns)
    :returns X_train: training data

    :rtype y_train: 1-dim array of length examples
    :returns y_train: training labels

    :rtype X_test: array of shape (examples, channels, rows, columns)
    :returns X_test: training data

    :rtype y_test: 1-dim array of length examples
    :returns y_test: training labels

    """

    # as the data is very large, let's try to use memory mapped numpy arrays
    # for the labels: pandas has a read csv file function


    return X_train, y_train, X_test, y_test