# Predicting-Movie-Genres-Based-on-Plot-Summary
Codes for my NLP class project at UMass-Amherst. This is the link to my project report: https://arxiv.org/abs/1801.04813

The data can be downloaded here: https://drive.google.com/open?id=18A6pnvINkztmUId01qjsEmI4x4ZOgOYS
The data file used for training the LSTM model include:
  1. "data/data.pkl": a dictionary with keys being "train" and "test" and values being DataFrames for train and test data respectively. Each DataFrame has field column "X", of which each element is plot summary's list of tokens' ids in the vocabulary, and a column "y", of which each element is a 1x20 binary vector representing the genres the corresponding movie belong to.
  2. "vocab060000.pkl": a dictionary that maps each token to an integer from 0 to 59999.
  3. "genres_info.pkl": a dictionary that maps each genre to an integer from 0 to 19.
  
Descripiton of the modules:
1. utils.py includes classes (different memory cells) to be used in lstm.py
2. eval_utils.py includes functions to be used for evaluation in train_lstm.py
3. lstm.py includes the LSTM class used in train_lstm.py.
4. test_neighbors.py find nearest neighbors to words using word embeddings learned by the lstm model.
   This is to evaluluate the quality of the embeddings learned by the LSTM network.
    
