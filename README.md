# seq2seq_chatbot_ch
a chinese chatbot with seq2seq model

# statement
使用TensorFlow实现的sequence to sequence 聊天机器人模型。本项目参考自 https://github.com/sea-boat/seq2seq_chatbot，
使用中文问答语料进行训练。训练语料为 https://github.com/candlewill/Dialog_Corpus 中的egret-wenda-corpus.zip

# use
1、运行data文件夹下的data_ch.py，对问答语料进行处理，用到了nltk词频统计和jieba分词
2、运行train.py训练模型。训练是一个耗时的过程，这里使用的训练数据只有2000个问题，训练5000个迭代却需要训练几个小时
3、运行test_model可以看到效果
