# -*- coding: UTF-8 -*-

'''
处理训练数据，数据中第一行为问题，第二行为回答，第三行又为问题，依次下去
'''

import nltk
import itertools
import numpy as np
import pickle
import jieba
import tensorflow as tf

FileName = "wenda.txt"

limit = {
    'maxq': 50,  # 问题最大长度
    'minq': 0,  # 最小长度
    'maxa': 48,  # 回答最大长度
    'mina': 3  # 回答最小长度
}

UNK = 'unk'
GO = '<go>'
EOS = '<eos>'
PAD = '<pad>'
VOCAB_SIZE = 5000  # 词汇数量


def cut_word(sentence):
    seg_list = jieba.cut(sentence)
    return tf.compat.as_str("/".join(seg_list)).split('/')


def process_data():
    qtokenized, atokenized = [], []
    lines = open(FileName, encoding='UTF-8').read().split('\n')
    data_len = len(lines)
    for i in range(0, data_len, 2):
        qline, aline = lines[i], lines[i + 1]
        qline, aline = aline.lower(), aline.lower()
        qList = qline.split("+++$+++")
        aList = aline.split("+++$+++")
        qline, aline = qList[-1][1:], aList[-1][1:]

        qWords, aWords = cut_word(qline), cut_word(aline)
        if limit['maxq'] >= len(qWords) and limit['minq'] <= len(qWords) and limit['maxa'] >= len(aWords) and limit[
            'mina'] <= len(aWords):
            qtokenized.append(qWords)
            atokenized.append(aWords)

    filter_len = len(qtokenized)
    filtered = 100-int(filter_len * 100 / (data_len // 2))
    print(str(filtered) + "% filtered from original data")
    idx2w, w2idx, freq_dist = index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)
    print(idx2w)
    with open("idx2w.pkl", 'wb') as f:
        pickle.dump(idx2w, f)
    with open('w2idx.pkl', 'wb') as f:
        pickle.dump(w2idx, f)
    idx_q, idx_a, idx_o = zero_pad(qtokenized, atokenized, w2idx)

    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)
    np.save('idx_o.npy', idx_o)

    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def index_(tokenized_sentences, vocab_size):
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    index2word = [GO] + [EOS] + [UNK] + [PAD] + [x[0] for x in vocab]
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index, freq_dist


def zero_pad(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)
    # +2 dues to '<go>' and '<eos>'
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)
    idx_o = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'], 1)
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 2)
        o_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 3)
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)
        idx_o[i] = np.array(o_indices)

    return idx_q, idx_a, idx_o


def pad_seq(seq, lookup, maxlen, flag):
    if flag == 1:
        indices = []
    elif flag == 2:
        indices = [lookup[GO]]
    elif flag == 3:
        indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    if flag == 1:
        return indices + [lookup[PAD]] * (maxlen - len(seq))
    elif flag == 2:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (maxlen - len(seq))
    elif flag == 3:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (maxlen - len(seq) + 1)


if __name__ == '__main__':
    process_data()
