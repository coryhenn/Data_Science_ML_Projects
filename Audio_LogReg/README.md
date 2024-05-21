# Purpose
This model implements custom-built logistic regression using softmax predictions to classify audio clips into genres using labeled audio files. 

# How to Run the Code
IMPORTANT: Sample Dataset MUST be inside following path: **'./data/train/<genre>/*.au'** 
Test Dataset MUST be inside following path: **'./data/test/*.au'** 
Python programs must be in the current directory: **'./<python files>'**

Follow the steps below in the order given:
Step 1.) Load and sample all the audio files in each genre using prepareDataFile.py, Run prepareDataFile.py with the path of train data directory.  
Usage: **'python prepareDataFIle.py ./data/train'**
creates a **'train_data.csv'** file in current directory using the all ther genres subdirectory and audio file in each genre. 

Step 2.) To create a **'test_data.csv'** file in current directory, run the following code: 
Usage: **'python prepare_testdata.py ./data/test'**

Step 3.) Run softmax based logistic regression using, <> is a required argument, [] are optional arguments, default value will be used (1000, 0.1, 0.5) 
Usage: **'python logisticRegressionBySoftmax.py  [iterations] [learning rate] [threshold]'** 
Example 1: **'python .\logisticRegressionBySoftmax.py .\train_data.csv'** 
Example 2: **'python .\logisticRegressionBySoftmax.py .\train_data.csv 5000 0.01'** 
Example 3: **'python .\logisticRegressionBySoftmax.py .\train_data.csv 5000 0.01 0.5'**

# File manifest :: description of each file

**'prepareDataFile.py'** - Reads audio files from ./data/train folder. Reads all the *.au files from each genres sub-folder and output a CSV file (./train_data.csv) with samples comtaining features and target in the current directory.

**'prepare_testdata.py'** - Reads test audio files from ./data/test folder. Reads all the audio file and output a CSV file (./test_data.csv) for testing in current directory. 

**'logisticRegressionBySoftmax.py'** - Multiclass logistic regression using Softmax function. It outputs accuracy. 
