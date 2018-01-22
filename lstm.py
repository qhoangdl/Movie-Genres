"""
This module include the LSTM class. The class builds a LSTM network according to some arguments:
    - batch_size: the size of mini-batches for gradient descent training
    - state_size: the number of hidden units per layer
    - num_layers: the number of layers in the network
    - num_genres: the number of genres for the task
    - num_epochs: the number of epoch for training.
    - learning_rate: the learning rate for Adam optimizer
    - keep_prob: the drop out keep rate
    - method: the method used for genre prediction.
              "softmax" is for the multinomial approach,
              "rank" is for the rank method
              "sigmoid" is for the k-binary transformation approach,
              "wsigmoid" is similar to "sigmoid" but adjust for the weight of different genres
    - cell_type: the type of LSTM cell to be used. It can be "GRU", "LNGRU", "LNGRU2" or "LNLSTM"
    - clipping_threshold: the threshold value for global gradient clipping.

Some important functions are:
    - fit(df): takes a DataFrame df as train data and train the network. df has two series. df["X"]
               is the input sequence of word's ids in the vocabulary. df["y"] is a 1x20 binary array
               with each element is 1 if the corresponding genre is in the true genres.
    - predict(seq, convert=False, tokenize=False): make prediction
               args:
                    seq: a list of sequences. The sequences can be a string or a list of words or integers
                    convert: if true, then seq needs to be convert into list of ids in the vocabulary
                    tokenize: if true, then each element in seq must be tokenize.
                return:
                    a list of network outputs for each element in seq. The output for each element is an
                    array of num_genres element. The output can means
                        - the conditional probability of each genre if the method is "softmax"
                        - the rank value if the method is "rank"
                        - the logit output for each genre if the method is "sigmoid" or "wsigmoid". positive
                          logit means positive.
    - neighbors(words, k): return the top k nearest words based on cosine distance for each word in
               the list words. words can be a list of string or a list of ids in the vocabulary.

"""

import numpy as np
import tensorflow as tf
import os
import dill as pkl
from nltk.tokenize import word_tokenize
from utils import *

class LSTM:
    def __init__(self,
                 model_name='LSTM',
                 batch_size=128,
                 state_size=128,
                 num_layers=2,
                 vocab_size=60000,
                 num_genres=20,
                 num_epochs=200,
                 learning_rate=0.001,
                 alpha=0.0,
                 keep_prob=0.90,
                 method="softmax",
                 cell_type="GRU",
                 activation=tf.nn.tanh,
                 clipping_threshold=None,
                 vocab_dict_fp="data_lstm/vocab060000.pkl",
                 genres_dict_fp="data_lstm/genres_info.pkl",
                 log_path="logs/",
                 model_path="model/lstm.ckpt",
                 summary_freq=None,
                 random_state=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.num_genres = num_genres
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.keep_prob_rate = keep_prob
        self.method = method
        self.cell_type = cell_type
        self.activation = activation
        self.clipping_threshold = clipping_threshold
        vocab_dict = pkl.load(open(vocab_dict_fp, "rb"))
        genres_dict = pkl.load(open(genres_dict_fp, "rb"))
        self.vocab, self.vocab_id = vocab_dict["vocab"], vocab_dict["vocab_id"]
        self.genres, self.genres_id = genres_dict["genres"], genres_dict["genres_id"]
        self.log_path = log_path
        self.model_path = model_path
        self.summary_freq = summary_freq
        self.random_state = random_state

    def _tf_init(self):
        self.tf_graph = tf.Graph()
        self.tf_config = get_default_config()
        self.tf_session = tf.Session(config=self.tf_config, graph=self.tf_graph)

        if self.random_state is not None:
            with self.tf_graph.as_default():
                tf.set_random_seed(self.random_state)

        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))

        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))

    def init_pos_weight(self, df):
        prob = df.mean()
        self.pos_weight = np.ones((self.batch_size, 1)) * (1 - prob) / prob
        self.genres_weight = 1 / (1 - prob)

    def _build_model(self):
        self.X = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='X')
        self.seqlen = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_genres], name='y')
        self.keep_prob = tf.placeholder(tf.float32)
        self.logits = self._create_rnn(self.X, self.seqlen, self.keep_prob, reuse=False, name="rnn")

        # make prediction
        self.pred = self._create_rnn(self.X, self.seqlen, self.keep_prob, reuse=True, name="rnn")
        if self.method == "softmax":
            self.pred = tf.nn.softmax(self.pred)

        # get top k nearest words
        self.word_list = tf.placeholder(tf.int32, shape=[None])
        self.k = tf.placeholder(tf.int32)
        self.nearest_values, self.nearest_indices = self._nearest(self.word_list, self.k, name="rnn")

        if self.method == "softmax":
            genres_per_example = tf.reduce_sum(self.y, axis=1, keep_dims=True)
            labels = self.y / genres_per_example
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels),
                name="softmax_loss")
        elif self.method == "sigmoid":
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y),
                name="sigmoid_cross_entropy"
            )
        elif self.method == "wsigmoid":
            self.loss = tf.reduce_mean(self.genres_weight * tf.nn.weighted_cross_entropy_with_logits(
                logits=self.logits, targets=self.y, pos_weight=self.pos_weight),
                name="weighted_cross_entropy"
            )
        else:
            self.loss = self._rank_loss(self.logits, self.y, name="rank_loss")

        self.opt = self._create_optimizer(self.loss, scope="rnn", lr=self.learning_rate)

    def _create_rnn(self, X, seqlen, keep_prob, reuse=False, name="rnn"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            # Embedding layer
            embeddings = tf.get_variable("embeddings", [self.vocab_size, self.state_size])
            rnn_inputs = tf.nn.embedding_lookup(embeddings, X)

            # RNN
            cells = []
            for i in range(self.num_layers):
                if self.cell_type == "GRU":
                    cell = tf.nn.rnn_cell.GRUCell(self.state_size)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
                elif self.cell_type == "LNGRU":
                    cell = LayerNormGRUCell(self.state_size, keep_prob=keep_prob,
                                            activation=self.activation, seed=self.random_state)
                elif self.cell_type == "LNGRU2":
                    cell = LNGRU2(self.state_size, keep_prob=keep_prob,
                                  activation=self.activation, seed=self.random_state)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
                else:
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.state_size, dropout_keep_prob=keep_prob)
                cells.append(cell)

            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            else:
                cell = cells[0]

            if self.cell_type == "GRU":
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            init_state = cell.zero_state(self.batch_size, tf.float32)
            outputs, state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen, initial_state=init_state)
            last_output = tf.gather_nd(outputs, tf.stack([tf.range(self.batch_size), seqlen - 1], axis=1))
            logits = linear(last_output, self.num_genres, scope="logits")
            tf.summary.histogram('rnn/logits/activations', logits)
            if self.cell_type == "LMN2":
                logits = layer_norm(logits, "logits/")
                tf.summary.histogram('rnn/logits/normalized', logits)
            return logits

    def _nearest(self, X, k, name="rnn"):
        with tf.variable_scope(name, reuse=True):
            embeddings = tf.get_variable("embeddings", [self.vocab_size, self.state_size])
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            X = tf.nn.embedding_lookup(embeddings, X)
            similarity = tf.matmul(X, normalized_embeddings, transpose_b=True)
            values, indices = tf.nn.top_k(similarity, k)
            return values, indices

    def _rank_loss(self, r, y, name="rank_loss"):
        r_vertical = tf.reshape(r, [self.batch_size, -1, 1])
        r_horizontal = tf.reshape(r, [self.batch_size, 1, -1])
        r_diff = tf.exp(r_vertical - r_horizontal)

        y_vertical = tf.reshape(1 - y, [self.batch_size, -1, 1])
        y_horizontal = tf.reshape(y, [self.batch_size, 1, -1])
        y_prod = y_vertical * y_horizontal

        num_pairs = tf.reduce_sum(y, axis=1) * tf.reduce_sum(1 - y, axis=1)
        loss_per_example = tf.reduce_sum(r_diff * y_prod, axis=[1, 2]) / num_pairs
        loss = tf.reduce_mean(loss_per_example, name=name)
        return loss

    def _create_optimizer(self, loss, scope, lr):
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5)
        grads = opt.compute_gradients(loss, var_list=params)
        if self.clipping_threshold is None:
            train_op = opt.apply_gradients(grads)
        else:
            gradients, variables = zip(*grads)
            gradients, _ = tf.clip_by_global_norm(gradients, self.clipping_threshold)
            train_op = opt.apply_gradients(zip(gradients, variables))


        for var in params:
            m = opt.get_slot(var, "m")  # get the first-moment vector
            v = opt.get_slot(var, "v")  # get the second-moment vector

            m_hat = m / (1 - opt._beta1_power)  #  bias correction
            v_hat = v / (1 - opt._beta2_power)  # bias correction

            step = lr * m_hat / (v_hat**0.5 + opt._epsilon_t)  # update size
            update_ratio = tf.abs(step) / (tf.abs(var) + 1e-8)  # update ratio

            tf.summary.histogram(var.op.name + '/values', var)
            tf.summary.histogram(var.op.name + '/update_size', step)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        return train_op

    def fit_loop(self, train, val, num_epochs):
        current_epoch = 0
        while train.epochs < num_epochs:
            total_loss = 0
            for i in range(self.summary_freq-1):
                X, y, seqlen = train.next_batch(self.batch_size)
                loss, opt = self.tf_session.run([self.loss, self.opt],
                                                feed_dict={self.X: X, self.seqlen: seqlen, self.y: y + self.alpha,
                                                           self.keep_prob: self.keep_prob_rate})
                total_loss += loss

            loss, opt, summaries = self.tf_session.run([self.loss, self.opt, self.tf_merged_summaries],
                                                        feed_dict={self.X: X, self.seqlen: seqlen,
                                                                   self.y: y + self.alpha,
                                                                   self.keep_prob: self.keep_prob_rate})
            total_loss += loss
            current_epoch += 1
            self.tf_summary_writer.add_summary(summaries, current_epoch)
            loss_summary = tf.Summary(value=[
                tf.Summary.Value(tag="train_loss", simple_value=total_loss/self.summary_freq)
            ])
            self.tf_summary_writer.add_summary(loss_summary, current_epoch)
            self.file.write("Epoch {:05d}, Train Loss: {:.2f}\n".format(current_epoch, total_loss / self.summary_freq))
            print("Epoch {:05d}, Train Loss: {:.2f}".format(current_epoch, total_loss / self.summary_freq))

            step, total_loss = 0, 0
            val_epochs = val.epochs
            while val.epochs == val_epochs:
                step += 1
                X, y, seqlen = val.next_batch(self.batch_size)
                loss = self.tf_session.run(self.loss, feed_dict={self.X: X, self.seqlen: seqlen,
                                                                 self.y: y + self.alpha,
                                                                 self.keep_prob: 1.0})
                total_loss += loss

            loss_summary = tf.Summary(value=[
                tf.Summary.Value(tag="test_loss", simple_value=total_loss / step)
            ])
            self.tf_summary_writer.add_summary(loss_summary, current_epoch)
            self.file.write("Epoch {:05d}, Test  Loss: {:.2f}\n".format(current_epoch, total_loss / step))
            print("Epoch {:05d}, Test  Loss: {:.2f}".format(current_epoch, total_loss / step))

            if self.min_loss is None or self.min_loss > total_loss / step:
                self.min_loss = total_loss / step
                self.save(self.model_path)

    def fit(self, df):
        df["length"] = df["X"].apply(np.size)
        N = len(df)
        if self.summary_freq is None:
            self.summary_freq = (N // 10) // self.batch_size
        self.init_pos_weight(df["y"][:N // 10 * 9])
        train = BucketedDataIterator(df.iloc[:N // 10 * 9])
        val = BucketedDataIterator(df.iloc[N // 10 * 9:])
        self._tf_init()
        self._on_train_begin()
        self.fit_loop(train, val, self.num_epochs)
        self._on_train_end()

    def predict(self, seq, convert=False, tokenize=False):
        if len(seq) == 0:
            return 0

        data = seq
        if convert:
            data = []
            for x in seq:
                if tokenize:
                    x = word_tokenize(x.lower())
                data.append(np.array([self.vocab_id.get(word, 1) for word in x] + [0], dtype=np.int32))

        res = np.zeros(((((len(seq) - 1) // self.batch_size + 1) * self.batch_size), self.num_genres))
        for i in range(0, len(seq), self.batch_size):
            seqlen = np.zeros(self.batch_size, dtype=np.int32)
            for j in range(self.batch_size):
                if (i + j < len(data)):
                    seqlen[j] = data[i + j].shape[0]

            maxlen = np.max(seqlen)
            X = np.zeros((self.batch_size, maxlen), dtype=np.int32)
            for j, X_j in enumerate(X):
                if (i + j < len(data)):
                    X_j[:seqlen[j]] = data[i + j]

            res[i : i + self.batch_size] = self.tf_session.run(self.pred, feed_dict={self.X: X, self.seqlen: seqlen,
                                                                                     self.keep_prob: 1.0})

        return res[:len(seq)]

    def neighbors(self, words, k=1):
        if len(words) == 0:
            return 0, 0

        X = np.zeros(len(words))
        if type(words[0]) == str:
            for i, word in enumerate(words):
                X[i] = self.vocab_id.get(word, 1)

        values, indices = self.tf_session.run([self.nearest_values, self.nearest_indices],
                                              feed_dict={self.word_list: X, self.k: k})

        values_dict = {}
        neighbor_dict = {}
        for i, word in enumerate(words):
            values_dict[word] = values[i]
            neighbor_dict[word] = [self.vocab[indices[i, j]] for j in range(k)]

        return values_dict, neighbor_dict

    def _on_train_begin(self):
        with self.tf_graph.as_default():
            self._build_model()
            # merge all summaries
            self.tf_merged_summaries = tf.summary.merge_all()
            # Initialize all variables
            self.tf_session.run(tf.global_variables_initializer())
        self.min_loss = None
        self.tf_summary_writer = tf.summary.FileWriter(self.log_path, self.tf_session.graph)
        self.file = open(os.path.join(self.log_path, "progress.txt"), "w+")

    def _on_train_end(self):
        self.tf_summary_writer.close()
        self.file.close()

    def _get_session(self):
        return self.tf_session

    def save(self, file_path="model/lstm.ckpt"):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        self._save_model(file_path)
        return file_path

    def _save_model(self, file_path="model/lstm.ckpt"):
        # with open(file_path + ".pkl", 'wb') as f:
        #    pkl.dump({'model': self}, f)
        with self.tf_graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.tf_session, file_path)

    def load_model(self, file_path):
        # model = pkl.load(open(file_path + ".pkl", 'rb'))['model']
        # model.tf_graph = tf.Graph()
        # model.tf_config = get_default_config()
        # model.tf_session = tf.Session(config=model.tf_config, graph=model.tf_graph)

        self._tf_init()
        with self.tf_graph.as_default():
            # tf.get_variable_scope().reuse_variables()
            self._build_model()
            # merge all summaries
            self.tf_merged_summaries = tf.summary.merge_all()
            saver = tf.train.Saver()
            saver.restore(self.tf_session, file_path)

