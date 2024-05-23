import torch
from torch import nn
import FeatureData as idc
import MLPModel as mlp
import numpy as np
import sys
from datetime import datetime
from matplotlib import pyplot as plt
# using SciKit-Learn, since Pytorch doesn't have a built-in confusion matrix metric
from sklearn.metrics import confusion_matrix

# ## Function: Logger - Logs the std output in a file for debugging
# class Logger(object):
#     def __init__(self, filename="logfile.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")

#     def write(self, output):
#         self.terminal.write(output)
#         self.log.write(output)

# # TODO: Comment out the next two line to save disk space. It is for debugging only
# current_datetime = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
# logfile = "runMLP-%s.log" % current_datetime
# print("runMLP: Log file: ", logfile)
# sys.stdout = Logger(logfile)
# ##


# Select loos function to calculate loss
lossFn = nn.CrossEntropyLoss()


def trainEach(model, dataLoader, lossFn, optimizer):
    for inputs, targets in dataLoader:
        inputs, targets = inputs.to("cpu"), targets.to("cpu")

        # loss
        predictions = model(inputs)
        loss = lossFn(predictions, targets)

        # backpropagation of loss and weights update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"runMLP: Loss: {loss.item()}")


def train(model, dataLoader, lossFn, optimizer, epochs):
    for i in range(epochs):
        print(f"runMLP: Epoch {i+1}")
        trainEach(model, dataLoader, lossFn, optimizer)
        print("runMLP: -------------------")
    print("runMLP: Training is done")


def trainIt(model, dataLoader, optimizer):
    model.train()
    train_loss = 0

    for batch, tensor in enumerate(dataLoader):
        data, target = tensor

        # feed forward
        optimizer.zero_grad()
        out = model(data)
        loss = lossFn(out, target)
        train_loss += loss.item()

        # backpropagate
        loss.backward()
        optimizer.step()

    # Return average loss
    avg_loss = train_loss / (batch + 1)
    print(
        'runMLP: runMLP: Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


def testIt(model, data_loader):
   # Switch the model to evaluation mode
    model.eval()

    test_loss = 0
    matched = 0

    # For test data, no need to use gradient, no Tensor.backward()
    with torch.no_grad():
        batch_count = 0
        for batch, tensor in enumerate(data_loader):
            batch_count += 1

            data, target = tensor
            out = model(data)

            test_loss += lossFn(out, target).item()

            _, predicted = torch.max(out.data, 1)
            matched += torch.sum(target == predicted).item()

    avg_loss = test_loss/batch_count

    print('runMLP: Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, matched, len(data_loader.dataset),
        100. * matched / len(data_loader.dataset)))

    # return average loss for the epoch
    return avg_loss


if __name__ == "__main__":

    NURONS = 256
    #BATCH_SIZE = 128
    BATCH_SIZE = 64
    #EPOCHS = 20
    EPOCHS = 100
    #LEARNING_RATE = 0.01
    LEARNING_RATE = 0.001

    class_mapping = ["blues", "classical", "country", "disco",
                     "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    model_file = "MLP.mod"

    inputData = idc.InputData()
    df, X, y = inputData.readfromFile("train_data.csv")

    inputData.features = X.shape[1]
    inputData.targets = inputData.targetClasses(y)
    INPUT_FEATURES = inputData.features
    TARGETS = inputData.targets

    # Input data information on raw data with features extraction from the audio data
    inputData.inputInfo(df)

    # Standardizing the data to fit all the features values within a managable range for non-linear functions.
    X = inputData.standardizeFeatures(df)

    # splitting the data in train and validatin test datasets
    X_train, X_test, y_train, y_test = inputData.splitDataset(X, y)

    train_dataLoader = inputData.toDataLoader(X_train, y_train, BATCH_SIZE)
    test_dataLoader = inputData.toDataLoader(X_test, y_test, BATCH_SIZE)

    # Setup the NN MLP model
    mlpObj = mlp.MLP(INPUT_FEATURES, NURONS, TARGETS)
    model = mlpObj.to("cpu")

    # Select the optimizer from pytorch
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad

    epoch_nums = []
    training_loss = []
    validation_loss = []

    print("runMLP: Epocs: ", EPOCHS)
    # Run through number of Epochs
    for epoch in range(1, EPOCHS + 1):
        # print the epoch number
        print(f"runMLP: Epoch: {epoch}")

        train_loss = trainIt(model, train_dataLoader, optimizer)
        test_loss = testIt(model, test_dataLoader)

        # keep track losses for each epoch run
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

    # Plot the training and validation losses characteristics
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()

    for param_tensor in model.state_dict():
        print("runMLP:", param_tensor, "\n",
              model.state_dict()[param_tensor].numpy())

    # Set the model to evaluate mode
    model.eval()

    # Get predictions for the test data
    X_test_tensor = inputData.X2Tensor(X_test)
    _, predicted = torch.max(model(X_test_tensor).data, 1)

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, predicted.numpy())
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_mapping))
    plt.xticks(tick_marks, class_mapping, rotation=45)
    plt.yticks(tick_marks, class_mapping)
    plt.xlabel("Predicted Genres")
    plt.ylabel("Actual Genres")
    plt.show()

    # Store the model for future use
    torch.save(model.state_dict(), model_file)
    del model
    print(f"runMLP: Model trained ans stored at {model_file}")
