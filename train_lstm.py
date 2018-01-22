"""
The main function train a model to predict movie genres based on plot summaries. It performs a few step:
    1. Train a softmax LSTM network to predict the probability of a movie belonging to each genre
    2. Train a probability threshold regressor
    3. Use the LSTM network and the probability threshold regressor to make predictions on the test examples
    4. Evaluate the predictions on the test examples
    5. Display the predictions on some random test examples
"""
from xgboost import XGBRegressor
from eval_utils import *
import tensorflow as tf
from lstm import LSTM
import numpy as np
import dill as pkl
import argparse
import sys
import os

FLAGS = None

def prob_threshold(prob, y):
    """
    :param prob: the categorical distribution
    :param y: a 1x20 binary array with each element is 1 if the corresponding genre is in the true genres.
    :return: the probability threshold
    """
    pairs = sorted(tuple(zip(prob, y)), key=lambda x: x[0])
    threshold = 0
    count = np.sum(1 - y)
    best = count

    for i in range(len(pairs) - 1):
        count += 1 if pairs[i][1] > 0.5 else -1

        if best > count:
            best = count
            threshold = (pairs[i][0] + pairs[i + 1][0]) / 2

    return threshold

def main(_):
    data_fp = "data/data.pkl"
    df = pkl.load(open(data_fp, "rb"))
    train_df = df["train"]

    model_name = "lstm_lngru_iden_softmax_l2s128kp0.8lr0.1"
    root_dir = os.path.join(os.getcwd(), "models", model_name)
    result_dir = os.path.join(os.getcwd(), "results", "{}.txt".format(model_name))
    model=LSTM(model_name=model_name,
               batch_size=FLAGS.batch_size,
               state_size=FLAGS.state_size,
               num_layers=FLAGS.num_layers,
               vocab_size=FLAGS.vocab_size,
               num_genres=FLAGS.num_genres,
               num_epochs=FLAGS.num_epochs,
               learning_rate=FLAGS.learning_rate,
               keep_prob=FLAGS.keep_prob,
               method=FLAGS.method,
               cell_type=FLAGS.cell_type,
               activation=None,
               clipping_threshold=FLAGS.clipping_threshold,
               random_state=FLAGS.random_state,
               vocab_dict_fp="data/vocab060000.pkl",
               genres_dict_fp="data/genres_info.pkl",
               log_path=os.path.join(root_dir, "logs/"),
               model_path=os.path.join(root_dir, "model/lstm.ckpt"))

    # fit model
    model.fit(train_df)
    # load the best model after training
    model.load_model(model.model_path)

    #  get the train/val/test dataset
    train_df = df["train"].iloc[:len(df["train"]) // 10 * 9].reset_index(drop=True).copy()
    val_df = df["train"].iloc[len(df["train"]) // 10 * 9:].reset_index(drop=True).copy()
    test_df = df["test"]

    # make prediction on the train/val/test dataset
    train_pred = model.predict(train_df["X"])
    val_pred = model.predict(val_df["X"])
    test_pred = model.predict(test_df["X"])

    # calculate the probability threshold on the train/val dataset
    train_threshold = np.zeros(len(train_df))
    for i in range(len(train_threshold)):
        train_threshold[i] = prob_threshold(train_pred[i], train_df["y"][i])

    val_threshold = np.zeros(len(val_df))
    for i in range(len(val_threshold)):
        val_threshold[i] = prob_threshold(val_pred[i], val_df["y"][i])

    # train a threshold regressor
    reg = XGBRegressor(max_depth=5, nthread=64, n_estimators=200, reg_lambda=1000, subsample=0.8, colsample_bytree=0.8,
                       seed=73)
    reg.fit(train_pred, train_threshold)
    pkl.dump({'model': reg}, open(os.path.join(root_dir, "model/reg.pkl"), "wb"))
    print("Train R-squared: {:.2f}".format(reg.score(train_pred, train_threshold)))
    print("Val R-squared: {:.2f}".format(reg.score(val_pred, val_threshold)))

    # make genre prediction
    test_threshold = reg.predict(test_pred)
    predict = []
    for i in range(len(test_pred)):
        genres = []
        for j in range(len(test_pred[i])):
            if test_pred[i][j] > test_threshold[i]:
                genres.append(model.genres[j])
        if len(genres) == 0:
            genres.append(model.genres[np.argmax(test_pred[i])])
        predict.append(genres.copy())

    # get the list of true genres
    test_genres = []
    for i, y in enumerate(test_df["y"]):
        genres = []
        for j in range(20):
            if y[j] > 0.5:
                genres.append(model.genres[j])
        test_genres.append(genres.copy())

    # evaluation
    if not os.path.exists(os.path.dirname(result_dir)):
        os.makedirs(os.path.dirname(result_dir))
    f = open(result_dir, "w")
    jaccards, TP, FP, TN, FN = eval(test_genres, predict, model.genres, f)
    predict_dist(predict, f)
    f.close()

    # show some test examples
    test_genres = open("data/test_genres.txt", encoding="utf-8").read().split('\n')
    test_plots = open("data/test_plots.txt", encoding="utf-8").read().split('\n')
    check_list = np.random.choice(20000, 10)
    seq = [test_plots[i] for i in check_list]
    seq_predict = model.predict(seq, convert=True, tokenize=True)
    seq_threshold = reg.predict(seq_predict)
    seq_genres = []
    for i in range(len(seq_predict)):
        genres = []
        for j in range(len(seq_predict[i])):
            if seq_predict[i][j] > seq_threshold[i]:
                genres.append(model.genres[j])
        seq_genres.append(genres.copy())

    for i in range(len(seq_genres)):
        print(seq[i])
        print(seq_genres[i])
        print(test_genres[check_list[i]])
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--state_size', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of Layers.')
    parser.add_argument('--vocab_size', type=int, default=60000,
                        help='Size of vocabulary.')
    parser.add_argument('--num_genres', type=int, default=20,
                        help='Number of genres.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='Dropout keep probability.')
    parser.add_argument('--clipping_threshold', type=float, default=10.0,
                        help='Gradient norm clipping.')
    parser.add_argument('--method', type=str, default="softmax",
                        help='Loss function (softmax or rank_loss).')
    parser.add_argument('--cell_type', type=str, default="LNGRU",
                        help='Cell type (GRU, LNGRU, LNGRU2 or LNLSTM).')
    parser.add_argument('--random_state', type=int, default=7337,
                        help='Random state.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
