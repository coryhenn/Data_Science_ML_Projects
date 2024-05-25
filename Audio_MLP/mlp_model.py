from torch import nn


class MLP(nn.Module):
    def __init__(self, inFeatures, neurons, outTargets):
        super().__init__()
        self.features = inFeatures
        self.neurons = neurons
        self.targets = outTargets

        # self.oneDim = nn.Flatten() # If needed to flatten multi-dimensional to one dimentional
        self.innerLayers = nn.Sequential(
            # if 2-D is 8x8, flatten will create 64 colms/features
            nn.Linear(inFeatures, neurons),
            nn.ReLU(),  # Activation Function
            nn.Linear(neurons, outTargets)
        )
        self.softmax = nn.Softmax(dim=1)

    # Allow us to tell pytorch, how to  process input data and how to manupulate data. in what sequence
    def forward(self, inputData):
        # oneDimX = self.oneDim(inputData) # if flatten is need, in our case, each autio data is already in one row
        #layerOut = self.innerLayers(oneDimX)
        layerOut = self.innerLayers(inputData)
        predictions = self.softmax(layerOut)
        return predictions
