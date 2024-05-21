import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

#                    0          1          2          3       4         5       6        7      8        9
Geners = np.array(['blues', 'classical', 'country', 'disco',
                  'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
GenreIndex = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

##
# soft_max() - Creates a probability matrix of N x K
# Params - z
# Return - 2-D matrix N x K where N is number of samples, K is number of classes


def soft_max(z):  # Tested againt scipy's softmax, same output
    # amax return the maximum of an array or maximum along an axis.
    # Here we are using along cols, so look at rowise maximum
    max_x = np.amax(z, axis=1).reshape(z.shape[0], 1)
    # using softmax formula
    ex = np.exp(z - max_x)
    return ex / ex.sum(axis=1, keepdims=True)

# Note: learning_rate is used as the step size when performing gradient descent (the optimization techniques)
# It’s important to choose a reasonable learning_rate, since if learning_rate is too small, then it takes much longer to reach the lower point,
# but if learning_rate is too large, then maybe we find us at a even higher loss point after taking the jump.
##
# gradient_descent() -
# Parameters: X - feature samples, y - target, iteration - number of iteration to adjust weight matrix,
#             learning_rate or alpha - multiplication factor of gradient, mu - for loss regularization
# Return - Adjusted Weight matrix


def gradient_descent(X, y, num_iterations=1000, learning_rate=0.1, mu=0.01):
    loss_list = []
    accuracy_list = []

    # For multi-class logistic regression, use onehot matrix of N x K
    y_onehot = to_one_hot(y)

    # initialize weights
    W = np.zeros((X.shape[1], y_onehot.shape[1]))

    print("Iteration:", num_iterations,
          "Learing rate:", learning_rate, "mu:", mu)
    for i in range(num_iterations):
        #print("W shape: ", W.shape, "\nW:", W)

        # Calculate gradient
        dw = calculate_gradient(X, y_onehot, W, mu)

        # Reduce weight byt the learning_rage and gradient
        W = W - (learning_rate * dw)

        predicted = predict(X, W)
        accuracy_list.append(accuracy(predicted, y))

        # Calculate loss based on the new weight matrix
        loss_list.append(calculate_loss(X, y_onehot, W))

    return W, loss_list, accuracy_list

##
# Function: loss() - Calculate the loos function using the negative log-likelihood function and
#                    normalized it by the sample size
#
#
# To avoid overfitting, it’s important to choose the right number of parameters for the model.
# This can be done using techniques like cross-validation or regularization. Cross-validation involves
# splitting the data into training and validation sets and testing the model on the validation set.
# Regularization involves adding a penalty term to the NLL function to discourage the model from fitting
# the data too closely.


def calculate_loss(X, y_onehot, W):
    weighted_X = - X @ W
    NbyN_diagonal_sum = np.trace(np.dot(np.dot(X, W), y_onehot.T))
    #print("NbyN_diagonal_sum:", NbyN_diagonal_sum)
    sum_log_sum_exp = np.sum(np.log(np.sum(np.exp(weighted_X), axis=1)))
    #print("sum_log_sum_exp:", sum_log_sum_exp)
    loss = (1/X.shape[0]) * (NbyN_diagonal_sum + sum_log_sum_exp)
    #print("loss:", loss)
    return loss

##
# calculate_gradient() - Calculate the gradient for weight update factor
# Params - X - Features sample data, y_onehot - NxK matrix of y, W - weight matrix,
#               mu - constant, if we want to use regularizaiton term
# Return gradient


def calculate_gradient(X, y_onehot, W, mu):
    weighted_X = - X @ W
    porobabilities = soft_max(weighted_X)
    dw = (1/X.shape[0]) * np.dot(X.T, (y_onehot - porobabilities)) + 2 * mu * W
    return dw

##
# Functions: to_one_hot()
# Param: y
# Return: a N x K matrix where N = number of samples, K = number of classes with 1 in each row to represent the class


def to_one_hot(y):  # tested against sklearn's OneHotEncoder
    classes, each_class_counts = np.unique(y, return_counts=True)
    y_count = len(y)
    class_count = len(classes)
    # Get NxK zero matrix
    one_hot = np.zeros((y_count, class_count))
    # Assign 1 to represent genre for a sample
    #print("y count:", y_count, "Classes:", classes)
    #print("y:", y)
    #print("one_host shape:", one_hot.shape)
    for i in range(y_count):
        #print("i:", i, "y[i]:", y[i])
        one_hot[i][y[i]] = 1
    return one_hot

##
# Functions: add_ones_for_bias() - add_ones_for_bias - add one's to the 1st column of samples, so that bias can be added wx+w0
# Param: X
# Return: training data with inserted bias


def add_ones_for_bias(X):
    return np.insert(X, 0, 1, axis=1)

#
##
# Functions: init_weights() - Initialize a W matrix with F = Number of Features, C = number of Class
# Param: F, C
# Return: initialize matrix with 0's


def init_weights(F, C):
    return np.zeros((F, C))

##
# Functions: add_bias_to_weight()
# Param: W, bias
# Return: a W with added bias


def add_bias_to_weight(W, bias):
    return np.insert(W, 0, bias, axis=0)

##
# Functions: predict()
# Param: X, W
# Return: predicted


def predict(X, W):
    Z = - np.dot(X, W)
    prob_matrix = soft_max(Z)
    # argmax returns the indices of the maximum values along an axis.
    # we are taking index of the maximum probabiliy in a row, so it will return index which has max probabilty and index also represent the class.
    # index represent the class catagor of a genres, it will return the predicted classes for samples
    predicted = np.argmax(prob_matrix, axis=1)
    return predicted

##
# Functions: accuracy()
# Param: predicted, actual
# Return: accuracy


def accuracy(predicted, actual):
    accuracy_array = predicted == actual
    yes, no = pd.DataFrame(accuracy_array).value_counts()
    accuracy = yes/(yes+no)
    #print("Accuray:", accuracy)
    return accuracy

##
# Function: conver_to_genres() - this function will convert target class value to genre name
# Params: sigmoid_predictions
# Return: pred_col


def conver_to_genres(sigmoid_predictions):
    pred_col = []
    rows = len(sigmoid_predictions)
    print("Row:", rows)
    for row in range(rows):
        pred_col.append(Geners[sigmoid_predictions[row]])
        #print(sigmoid_predictions[row], Geners[sigmoid_predictions[row]])
    return pred_col

##
# Function: LG_Sftmax() - this function will output accuracy using softmax model
# Params: X_train, y_train, class_count, num_iterations, learning_rate, mu
# Return: accuracy, optimized_W


def LG_Sftmax(X_train, y_train, class_count, num_iterations, learning_rate, mu):
    print("\nLogistic Regression using Softmax function ...")
    # Logistic Regression using softmax
    # initialize weights
    W = init_weights(X_train.shape[1], class_count)
    #print("W shape:", W.shape)

    # adding bias (b), W0
    W = add_bias_to_weight(W, 0.02)
    #W = add_bias_to_weight(W, 0.02)
    #print("W with row bias shape:", W.shape)

    #print("X_train shape:", X_train.shape)
    # Add W0 for each feature rows
    X_train = add_ones_for_bias(X_train)
    #print("X_train with 1st col ones shape:", X_train.shape)
    #print("X_train shape:", X_train)

    # optimized_W is the logistic model. Use this model to predic targets for sample data
    optimized_W, loss_list, accuracies = gradient_descent(
        X_train, y_train, num_iterations, learning_rate, mu)

    # Predicted classes for X_train usign optimized weights
    predicted_classes = predict(X_train, optimized_W)
    accuracy = predicted_classes == y_train
    yes, no = pd.DataFrame(accuracy).value_counts()
    accuracy = yes/(yes + no)
    print("Train data Accuray:", yes/(yes+no))
    return accuracy, optimized_W

##
# Function:  main() driving the softmax based multi-class logistic regression
# Argument: Training file


def main():
    num_iterations = 1000
    learning_rate = 0.01
    threshold = 0.5
    mu = 0.01
    print("\nDefaults without command line entry: Number of Iterations:", num_iterations,
          "Learning Rate:", learning_rate, "Threshold:", threshold, "Mu:", mu)
    cmdlineargs = len(sys.argv)
    if cmdlineargs < 2:
        print(
            "Usage ", sys.argv[0], "<Training data file> [iterations] [learning rate] [threshold] [mu]")
        print("Example:", sys.argv[0], "./train_data.csv")
        print("Etc. Please try again ....")
        sys.exit()
    if cmdlineargs > 2:
        num_iterations = int(sys.argv[2])
    if cmdlineargs > 3:
        learning_rate = float(sys.argv[3])
    if cmdlineargs > 4:
        threshold = float(sys.argv[4])
    if cmdlineargs > 5:
        mu = float(sys.argv[5])

    # Default value
    train_data = sys.argv[1]
    test_data = 'test_data.csv'

    if not os.path.exists(train_data):
        print("{} file does not exits.".format(train_data))
        sys.exit()

    print("Training Data File: ", train_data)
    print("Test Data File: ", test_data)

    # reading train dataset from csv
    df = pd.read_csv(train_data)
    print(df.head())
    print(df.info())
    print(df.describe())

    tdf = pd.read_csv(test_data)

    y = df['target']

    X = df.iloc[:, :-1].values
    y = df.target.values

    # id column of test output CSV file
    id = tdf.id.values
    X_testdata = tdf.iloc[:, :-1].values

    # Find the number of classes in the target column/data
    classes, _ = np.unique(y, return_counts=True)
    print("Classes:", classes)
    class_count = len(classes)
    print("Class Count:", classes, class_count)

    # normalizing, without normalization, exponent of a big value can overflow to infinity
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(df.iloc[:, :-1], dtype=float))
    X_testdata = scaler.fit_transform(np.array(tdf.iloc[:, :-1], dtype=float))
    #print("X: ", pd.DataFrame(X).to_string())

    # spliting of dataset into train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("X_train shape:", X_train.shape)
    print("\nX_test shape:", X_test.shape)
    print("\ny_train shape:", y_train.shape)
    print("\ny_test shape:", y_test.shape)

    # Testing using 'LogisticRegression' libs
    print("Testing data using Logistic Regression libs...")
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    train_accuracy = clf.score(X_train, y_train)
    print('Testing the Accuracy in percentage calculated on the Train Data using sklearn Library is '+str(train_accuracy*100))
    test_accuracy = clf.score(X_test, y_test)
    print('Testing the Accuracy in percentage calculated on the Test Data using sklearn Library is '+str(test_accuracy*100))

    accuracy, optimized_W = LG_Sftmax(
        X_train, y_train, class_count, num_iterations, learning_rate, mu)

    # Predicted classes for X_test usign optimized weights
    X_test = add_ones_for_bias(X_test)
    predicted_classes = predict(X_test, optimized_W)
    #print("Predicted:", predict_probs.shape, "\nPrdict matrix:", pd.DataFrame(predicted_classes).to_string())
    accuracy = predicted_classes == y_test
    yes, no = pd.DataFrame(accuracy).value_counts()
    print("Test data Accuray:", yes/(yes+no))

    # Find prediction for X_test and then use it to find accuracy with know known y_test
    # Now use Optimized Ws from training data for each classes to predic targets
    X_testdata = add_ones_for_bias(X_testdata)
    predicted_classes = predict(X_testdata, optimized_W)
    softmax_predictions = conver_to_genres(predicted_classes)
    # Creating output CSV file from test data
    data = {"id": id, "class": softmax_predictions}
    #data = {"id": id,"class": None }
    # Convert NumPy array to DataFrame
    pdf = pd.DataFrame(data)
    # print(pdf.to_string())

    # putting predictions for transaction ids in a output file in CSV format with headers
    pridicted_by_test_data = "data/softmax_prdicted-%s.csv" % current_datetime
    pdf.to_csv(pridicted_by_test_data, index=False)


# __name__ is a special variable whose value is '__main__' if the module is being run as a script,
# and the module name if it's imported.
if __name__ == "__main__":
    main()
