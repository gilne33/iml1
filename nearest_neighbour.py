import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    classifier = {'k_value' : k, 'x_train' : x_train, 'y_train' : y_train} # a dictionary with the rellevant data
    return classifier

def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    k_val = classifier['k_value']
    x_train = classifier['x_train']
    y_train = classifier['y_train']

    predictions = []
    for xi in x_test:
        distances = [distance.euclidean(xi, xj) for xj in x_train]
        closet_indices = np.argsort(distances)
        k_nearest = closet_indices[:k_val]
        k_labels = y_train[k_nearest].astype(int)
        predicted_label = np.bincount(k_labels).argmax()
        predictions.append(predicted_label)
    return np.array(predictions).reshape(-1, 1)


    






def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def mytest():
    k = 1
    x_train = np.array([[1,2], [3,4], [5,6]])
    y_train = np.array([1, 0, 1])
    classifier = learnknn(k, x_train, y_train)
    x_test = np.array([[10,11], [3.1,4.2], [2.9,4.2], [5,6]])
    y_testprediction = predictknn(classifier, x_test)
    print(y_testprediction)

def task2():
    sampel_sizes = [1,10,25,50,75,100]
    k =1
    data = np.load('mnist_all.npz')

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    average_errors = []
    min_errors = []
    max_errors = []

    for sampel_size in sampel_sizes:
        sampel_size_errors = []

        print(f"sample size is {sampel_size}")
        for i in range(10):

            x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], sampel_size)

            x_test, y_test = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], sampel_size)
            y_test = y_test.reshape(-1,1).astype(int)

            classifer = learnknn(k, x_train, y_train)

            preds = predictknn(classifer, x_test)
            error = np.mean(y_test != preds)
            sampel_size_errors.append(error)

        average_errors.append(np.mean(sampel_size_errors))

        min_error = np.min(sampel_size_errors)
        max_error = np.max(sampel_size_errors)
        min_errors.append(min_error)
        max_errors.append(max_error)
        print(f"Average error: {np.mean(sampel_size_errors)}, Min error: {min_error}, Max error: {max_error}")

    plt.errorbar(
    sampel_sizes,
    average_errors,
    yerr=[np.array(average_errors) - np.array(min_errors), np.array(max_errors) - np.array(average_errors)],
    fmt='o-',
    capsize=5,
    label='average test error with range of error')

    plt.xlabel('Training Sample Size')  
    plt.ylabel('Average Test Error')  
    plt.title('Errors Plot')  
    plt.grid(True) 
    plt.legend()
    plt.show()


    # plt.plot(sampel_sizes, average_errors, marker='o')
    # plt.xlabel('Training Sample Size')  
    # plt.ylabel('Average Test Error')  

    # # Add a title
    # plt.title('Errors Plot')  
    # plt.grid(True) 
    # plt.show()
    
    


if __name__ == '__main__':

    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()


    # mytest()
    # for i in range(5):

    #     np.random.seed(i)

    #     simple_test()

    task2()


