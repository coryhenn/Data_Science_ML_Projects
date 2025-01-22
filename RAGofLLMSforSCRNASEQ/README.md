# Purpose
This model allows for standard scRNAseq analysis workflow with the addition of cell type annotation using the GPT LLM with Retrieval Augmented Generation (RAG) of PubMed articles for enhanced accuracy.

# File manifest :: description of each file

**'DAVID.ipynb'** - .

**'MLPModel.py'** - Defines MLP Model.

**'prepareDataFile.py'** - Reads audio files from ./data/train folder. Reads all the *.au files from each genres sub-folder and output a CSV file (./train_data.csv) with samples comtaining features and target in the current directory.

**prepare_testdata.py'** - Reads test audio files from ./data/test folder. Reads all the audio file and output a CSV file (./test_data.csv) for testing in current directory. 

**'runMLP.py'** - Create and save MLP model using the training and validation features data.
