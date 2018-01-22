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
   
   Some important arguments for the LSTM class are:
     a. "method": the method used for multilabel. It can be
        - "softmax" for the multinomial-based approach
        - "rank" for the "rank" method
        - "sigmoid" for the k-binary trasnformation approach
        - "wsigmoid" is similar to "sigmoid" but adjust for the weight of different genres
     b. "cell_type": the type of LSTM cell to be used. It can be "GRU", "LNGRU", "LNGRU2" or "LNLSTM"
     
   Some important functions of the class are:
     a. fit(df): takes a DataFrame df as train data and train the network.
     b. predict(seq, convert=False, tokenize=False): make prediction
        seq: a list of sequences. The sequences can be a string or a list of words or integers
        convert: if true, then seq needs to be convert into list of ids in the vocabulary
        tokenize: if true, then each element in seq must be tokenize.
     c. neighbors(words, k): return the top k nearest words based on cosine distance for each word in the list words.
        words can be a list of string or a list of ids in the vocabulary.
        
4. test_neighbors.py find nearest neighbors to words using word embeddings learned by the lstm model.
   This is to evaluluate the quality of the embeddings learned by the LSTM network.
    
