# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
from word_id import Word_Id_Map
import jieba


class Seq2SeqModel(object):
    def __init__(self, hidden_size,layers,batch_size,seq_length,embedding_size,learning_rate,num_encoder_symbols,num_decoder_symbols,is_pred):
        self.hidden_size = hidden_size
        self.layers = layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.num_encoder_symbols = num_encoder_symbols
        self.num_decoder_symbols = num_decoder_symbols
        self.is_pred = is_pred

        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, self.seq_length])
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, self.seq_length])

        self.targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, self.seq_length])
        self.weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.seq_length])

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.layers)

        if is_pred=='false':
            self.feed_previous = False
        else:
            self.feed_previous = True
        results,states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            tf.unstack(self.encoder_inputs, axis=1),
            tf.unstack(self.decoder_inputs, axis=1),
            self.cell,
            self.num_encoder_symbols,
            self.num_decoder_symbols,
            self.embedding_size,
            feed_previous=self.feed_previous
        )

        logits = tf.stack(results, axis=1)


        if self.is_pred=='true':
            self.pred = tf.argmax(logits, axis=2)
        else:
            self.loss = tf.contrib.seq2seq.sequence_loss(logits, targets=self.targets, weights=self.weights)
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


    def train(self,sess,epochs):
        model_dir = './model'
        saver = tf.train.Saver
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        train_weights = np.ones(shape=[self.batch_size, self.seq_length], dtype=np.float32)

        epoch = 0
        while epoch < epochs:
            epoch = epoch + 1
            print("epoch:", epoch)
            train_x, train_y, train_target = loadQA()
            for step in range(len(train_x) // self.batch_size):
                print("step:", step)
                train_encoder_inputs = train_x[step * self.batch_size:step * self.batch_size + self.batch_size, :]
                train_decoder_inputs = train_y[step * self.batch_size:step * self.batch_size + self.batch_size, :]
                train_targets = train_target[step * self.batch_size:step * self.batch_size + self.batch_size, :]
                op = sess.run(self.train_op, feed_dict={self.encoder_inputs: train_encoder_inputs, self.targets: train_targets,
                                                   self.weights: train_weights, self.decoder_inputs: train_decoder_inputs})
                cost = sess.run(self.loss, feed_dict={self.encoder_inputs: train_encoder_inputs, self.targets: train_targets,
                                                 self.weights: train_weights, self.decoder_inputs: train_decoder_inputs})
                print("loss:", cost)
                step = step + 1
            if epoch % 100 == 0:
                saver.save(sess, model_dir + '/model.ckpt', global_step=epoch + 1)

    def pred(self,sess,qsentence):
        saver = tf.train.Saver
        model_dir = './model/'
        module_file = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, module_file)
        map = Word_Id_Map()
        encoder_input = map.sentence2ids(cut_word(qsentence))

        encoder_input = encoder_input + [3 for i in range(0, 50 - len(encoder_input))]
        encoder_input = np.asarray([np.asarray(encoder_input)])
        decoder_input = np.zeros([1, 50])
        print('encoder_input : ', encoder_input)
        print('decoder_input : ', decoder_input)
        pred_value = sess.run(self.pred, feed_dict={self.encoder_inputs: encoder_input, self.decoder_inputs: decoder_input})
        print(pred_value)
        sentence = map.ids2sentence(pred_value[0])
        print(sentence)


def loadQA():
    train_x = np.load('./data/idx_q.npy', mmap_mode='r')
    train_y = np.load('./data/idx_a.npy', mmap_mode='r')
    train_target = np.load('./data/idx_o.npy', mmap_mode='r')
    return train_x, train_y, train_target


def cut_word(sentence):
    sentence = sentence.lower()
    seg_list = jieba.cut(sentence)
    return tf.compat.as_str("/".join(seg_list)).split('/')