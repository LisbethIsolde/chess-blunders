# Analyzing Chess Blunders For Different Skill Levels

## Data Set
The original data was downloaded from https://database.lichess.org/. We used the data from January and Februrary of 2015. 

The preprocessed pickle file can be found [here](https://drive.google.com/drive/folders/1BYp4JrKgBz04OFB6qoopEy1FegTfAHpm?usp=sharing). This file is already stripped
of all non-evaluated games and unnecessary metadata, and the gameplay has been parsed into a list of moves and a list of blunders. When using the .pkl files instead of the raw data, line 111 in `main.py` can be commented out. This reduces the runtime of the program significantly, but still takes around 45 minutes. When using smaller PGN files (e.g. November+December 2013) the total runtime is around 5-10 minutes.
