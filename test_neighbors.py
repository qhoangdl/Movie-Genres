"""
The main function display the top 10 nearest neighbors of some words based on
the cosine distance between the word embeddings learned by the LSTM network
"""

import tensorflow as tf
from lstm import LSTM
import argparse
import sys
import os

FLAGS = None

def main(_):
    model_name = "lstm_lngru_iden_softmax_l2s128kp0.8lr0.1"
    root_dir = os.path.join(os.getcwd(), "models", model_name)
    model = LSTM(model_name=model_name,
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

    # load model
    model.load_model(model.model_path)

    words = ['arrogant', 'beautiful', 'cruel', 'happy', 'sad', 'magnificent', 'miniaturized', 'miniature', 'spaceship',
             'hang', 'manipulate', 'rob', 'escape', 'pessimistic', 'optimistic', 'violence', 'gangster', 'thief',
             'glory', 'spiderman', 'smile', 'western', 'eastern', 'batman', 'pirate', 'sky', 'captain', 'mountain',
             'dream', 'nirvana', 'cigarette', 'drunk', 'successful', 'home', 'wife', 'warrior', 'happy', 'war' 
             'river', 'love', 'hate', 'kill', 'cowboy', 'bear', 'king', 'robot', 'gladiator', 'earth', 'sunshine',
             'cloud', 'snow', 'wind', 'hero', 'adventure', 'journey', 'cruel', 'funny', 'attack', 'conserve',
             'preserve', 'shrek', 'shoot', 'princess', 'prince', 'mighty', 'mother', 'father', 'nostalgia', 'angel',
             'fire', 'lion', 'tiger', 'ape', 'monkey', 'queen', 'car']

    values_dict, neighbor_dict = model.neighbors(words, 11)
    for key, value in neighbor_dict.items():
        print("--------- words near {} are: ---------".format(key))
        print(value)
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
