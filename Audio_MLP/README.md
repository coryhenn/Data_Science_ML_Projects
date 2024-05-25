# Purpose
This multilayer perceptron model is intended to classify music genres based on labeled audio clips. 

# How to use:
IMPORTANT: Sample Dataset MUST be inside following path: **'./data/train/<genre>/*.au'** 
Test Dataset MUST be inside following path: **'./data/test/*.au'** 
Python programs must be in the current directory: **'./<python files>'**
Follow the steps below in the order given:
Step 1.a) Load and sample all the audio files in each genre using prepareDataFile.py, Run prepareDataFile.py with the path of train data directory.  
Usage: 'python prepareDataFIle.py ./data/train' 
creates a 'train_data.csv' file in current directory using the all ther genres subdirectory and audio file in each genre.

Step 1.b) Make sure that train_data.csv file has been created in current directory

Step 2.) To create a 'test_data.csv' file in current directory, run the following code: 
Usage: 'python prepare_testdata.py ./data/test'

Step 3.a) To run and save MLP NN model. MLP model will be saved as ./MLP.mod
Usage: 'python runMLP.py' 
Example 1: 'python .\runMLP.py' 

Step 3.b) To load MLP saved model and to generate predicted targets based on data/test/*.au files 
Usage: 'python MLPInference.py' 
Example 1: 'python MLPInference.py' 
Output File: Output file will be generated under ./data, Example: MLP-.csv

# File manifest :: description of each file

**'FeatureData.py'** - Extract features from audio files under ./data/train/.

**'MLPModel.py'** - Defines MLP Model.

**'prepareDataFile.py'** - Reads audio files from ./data/train folder. Reads all the *.au files from each genres sub-folder and output a CSV file (./train_data.csv) with samples comtaining features and target in the current directory.

**prepare_testdata.py'** - Reads test audio files from ./data/test folder. Reads all the audio file and output a CSV file (./test_data.csv) for testing in current directory. 

**'runMLP.py'** - Create and save MLP model using the training and validation features data.
