import itertools

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import tensorflow as tf
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell

from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.recurent_dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="recurent_dropout")

    def get_feed_dict(self, words, labels=None,
                      dropout=1.0, recurent_dropout=1.0):
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        feed[self.dropout] = dropout
        feed[self.recurent_dropout] = recurent_dropout

        return feed, sequence_lengths

    def get_recurrent_cell(self, num_units):
        if self.config.recurrent_cell_type is "norm_lstm":
            return lambda: LayerNormBasicLSTMCell(num_units=num_units, dropout_keep_prob=self.recurent_dropout)
        elif self.config.recurrent_cell_type is "lstm":
            return lambda: tf.nn.rnn_cell.LSTMCell(num_units=num_units)
        else:
            raise ValueError("Incorrect cell_type '" + str(self.config.cell_type) + "'")

    def add_dropout(self, cell, input_size):
        if self.config.recurrent_cell_type is "lstm":
            return tf.contrib.rnn.DropoutWrapper(cell=cell,
                                                 input_keep_prob=self.dropout,
                                                 output_keep_prob=self.dropout,
                                                 state_keep_prob=self.dropout,
                                                 variational_recurrent=True,
                                                 input_size=input_size,
                                                 dtype=tf.float32)
        else:
            return tf.contrib.rnn.DropoutWrapper(cell=cell,
                                                 input_keep_prob=self.dropout,
                                                 output_keep_prob=self.dropout)

    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            self.word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                cell_creator = self.get_recurrent_cell(self.config.hidden_size_char)

                cell_fw = cell_creator()
                cell_bw = cell_creator()

                cell_fw = self.add_dropout(cell_fw, self.config.dim_char)
                cell_bw = self.add_dropout(cell_bw, self.config.dim_char)

                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                self.word_embeddings = tf.concat([self.word_embeddings, output], axis=-1)

    def add_logits_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_creator = self.get_recurrent_cell(self.config.hidden_size_lstm)

            cells_fw = [cell_creator() for _ in range(self.config.number_of_layers)]
            cells_bw = [cell_creator() for _ in range(self.config.number_of_layers)]

            dropout_input_size = self.config.dim_word
            if self.config.use_chars:
                dropout_input_size += self.config.hidden_size_char * 2

            cells_fw[0] = self.add_dropout(cells_fw[0], dropout_input_size)
            cells_bw[0] = self.add_dropout(cells_bw[0], dropout_input_size)

            for layer in range(1, self.config.number_of_layers):
                dropout_input_size = self.config.hidden_size_lstm * 2
                cells_fw[layer] = self.add_dropout(cells_fw[layer], dropout_input_size)
                cells_bw[layer] = self.add_dropout(cells_bw[layer], dropout_input_size)

            output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=cells_fw,
                cells_bw=cells_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

        with tf.variable_scope("proj"):
            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            pred = tf.layers.dense(output, self.config.ntags)
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_pred_op(self):
        if self.config.use_crf:
            self.labels_pred, self.viterbi_score = tf.contrib.crf.crf_decode(
                self.logits, self.trans_params, self.sequence_lengths)
        else:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        self.sample_weight = tf.placeholder(dtype=tf.float32, shape=[None, None],
                                            name='sample_weights')
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        if self.config.lambda_regularization_loss > 0.0:
            self.loss += self.get_l2_regularization()

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def get_l2_regularization(self):
        for unreg in [tf_var.name for tf_var in tf.trainable_variables()
                      if ("bias" in tf_var.name.lower())]:
            print(unreg)

        l2 = self.config.lambda_regularization_loss * sum(tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables() if not ("bias" in tf_var.name.lower())
        )

        return l2

    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()
        self.add_pred_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.loss, self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words)
        labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

        return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (_, words_indexes, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words_indexes, labels,
                    self.config.dropout, self.config.recurent_dropout)

            _, train_loss, summary, learning_rate = self.sess.run(
                    [self.train_op, self.loss, self.merged, self.learning_rate], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss), ("learning rate", learning_rate)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        dev_metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in dev_metrics.items()])

        train_metrics2 = self.run_evaluate(train)
        msg2 = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in train_metrics2.items()])

        self.logger.info("train: " + msg2 + " dev: " + msg)

        return dev_metrics["f1"]

    def print_statistics(self, precision, recall, fscore, idx2Label):
        width = 20
        print("")
        print('{:<{width}} {:<{width}} {:<{width}} {:<{width}}'.format(
            "Label", "Precission", "Recall", "F1", width=width)
        )
        for i in range(len(precision)):
            print('{:<{width}} {:<{width}.2f} {:<{width}.2f} {:<{width}.2f}'.format(
                idx2Label[i].strip(), precision[i], recall[i], fscore[i], width=width)
            )

    def calculate_statistics(self, labels, predictions):
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_true=labels,
            y_pred=predictions,
        )

        return precision, recall, fscore

    def convert_labels_to_array(self, labels, predictions, sequence_lengths):
        predictions2 = []
        for prediction, length in zip(predictions, sequence_lengths):
            predictions2.append(prediction[:length])

        labels = list(itertools.chain.from_iterable(labels))
        predictions = list(itertools.chain.from_iterable(predictions2))

        return (labels, predictions)

    def predict_labels(self, data):
        for words, words_indexes, labels in minibatches(data, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words_indexes)

            for lab, lab_pred, word, length  in zip(labels, labels_pred, words,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]

                lab_chunks      = set(get_chunks(lab, self.idx_to_tag, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred, self.idx_to_tag, self.config.vocab_tags))

        return (lab_chunks, lab_pred_chunks)

    def run_evaluate(self, test):
        all_predictions = []
        all_labels = []
        all_sequence_lengths = []

        for words, words_indexes, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words_indexes)
            all_predictions.extend(labels_pred)
            all_labels.extend(labels)
            all_sequence_lengths.extend(sequence_lengths)

        labels, predictions = self.convert_labels_to_array(all_labels, all_predictions, all_sequence_lengths)
        precision, recall, f1score = self.calculate_statistics(labels, predictions)

        precision = list(precision)
        recall = list(recall)
        f1score = list(f1score)

        for score in (precision, recall, f1score):
            del score[self.config.vocab_tags["O"]]

        accuracy = accuracy_score(labels, predictions)
        self.print_statistics(precision, recall, f1score, self.idx_to_tag)

        return {"f1": np.mean(f1score) * 100, "acc": accuracy * 100}


    def predict(self, words_raw):
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
