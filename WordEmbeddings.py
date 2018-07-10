import ntpath,os
import multiprocessing

import logging
import re
import time

from gensim.corpora import WikiCorpus
from gensim.models.phrases import Phraser, Phrases

from joblib import Parallel, delayed

import numpy as np
import tensorflow as tf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class WordEmbeddings(object):
    def __init__(self, inp, out, model_name="ft_sg",
                 window=10, min_count=3,size=300,
                 workers=multiprocessing.cpu_count(),
                 sample=0.001,
                 negative=5,
                 seed=48,
                 tf=False,
                 gpu=False):

        self.model_name = model_name
        self.tf = tf
        self.gpu = gpu
        if self.tf:
            import tensorflow as tf

            self.sentences = None
            self.vocab = None
            self.word2int = None
            self.int2word = None
            self.vocab_size = 0
            self.vectors = None

        if "w2v" in model_name:
            from gensim.models.word2vec import Word2Vec
            self.model = Word2Vec
        else:
            from gensim.models.fasttext import FastText
            self.model = FastText

        if "sg" in model_name:
            self.sg = 1
        else:
            self.sg = 0

        self.out = out
        self.inp = inp

        self.window = window
        self.size = size
        self.min_count = min_count
        self.workers = workers
        self.sample = sample
        self.negative = negative
        self.seed = seed

    @staticmethod
    def _sentence_to_wordlist(raw: str):
        return re.sub("[^a-zA-Z]", " ", raw).split()

    def clean_wiki(self, output_path="data"):
        wiki = WikiCorpus(self.inp, lemmatize=False, dictionary={})
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, ntpath.basename(self.inp)+"_clean"), 'w') as output:
            n = 0
            for i, text in enumerate(wiki.get_texts()):
                # Note: if you're using Python 2, use:
                # output.write(" ".join(unicode(text)) + "\n")
                output.write(" ".join(text) + "\n")
                if i > 0 and i % 10000 == 0:
                    print("Saved {0:,} articles".format(i))
                n = i
        print("Finished saving {0:,} articles".format(n))

        return list(wiki.get_texts())

    def get_sentences(self):
        wiki = WikiCorpus(self.inp, lemmatize=False, dictionary={})

        tokenized_sentences = list(wiki.get_texts())

        phrases = Phrases(tokenized_sentences)
        bigram = Phraser(phrases)

        sentences = list(bigram[tokenized_sentences])

        self.sentences = sentences

        return sentences

    def one_hot(self, word_index, vocab_size):
        vec = np.zeros(vocab_size)
        vec[word_index] = 1
        return vec

    def build_vocab(self, sentences=None):
        if sentences is None:
            if self.sentences is None:
                sentences = self.get_sentences()
            else:
                sentences = self.sentences
        words = set()
        word2int = dict()
        int2word = dict()

        for sent in sentences:
            for w in sent:
                words.add(w)

        for i, w in enumerate(words):
            word2int[w] = i
            int2word[i] = w

        self.vocab = words
        self.word2int = word2int
        self.int2word = int2word
        self.vocab_size = len(words)
        return words, word2int, int2word

    @staticmethod
    def _(word, nb_word):
        return [word, nb_word]

    def get_XY(self, sentences, word2int):
        data = Parallel(n_jobs=-1)(delayed(self._)(word, nb_word) for sentence in sentences
                                   for i, word in enumerate(sentence)
                                   for nb_word in
                                   sentence[max(i - self.window, 0):min(i + self.window, len(sentence)) + 1] if
                                   word != nb_word)

        x_train = Parallel(n_jobs=-1)(delayed(self.one_hot)(word2int[word[0]], self.vocab_size) for word in data)
        y_train = Parallel(n_jobs=-1)(delayed(self.one_hot)(word2int[word[1]], self.vocab_size) for word in data)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        return x_train, y_train


    def _tf_run(self, x_train, y_train):

        x = tf.placeholder(tf.float32, shape=(None, self.vocab_size))
        y_label = tf.placeholder(tf.float32, shape=(None, self.vocab_size))


        W1 = tf.Variable(tf.random_uniform([self.vocab_size, self.size], -1.0, 1.0))
        b1 = tf.Variable(tf.random_normal([self.size]))
        h = tf.add(tf.matmul(x, W1), b1)

        W2 = tf.Variable(tf.random_normal([self.size, self.vocab_size]))
        b2 = tf.Variable(tf.random_normal([self.vocab_size]))
        pred = tf.nn.softmax(tf.add(tf.matmul(h, W2), b2))

        device = "/gpu" if self.gpu else "cpu"
        with tf.device(device):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                sess.run(tf.global_variables_initializer())
                cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(pred), reduction_indices=[1]))
                train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

                for i in range(1000):
                    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
                    if i % 100 == 0:
                        print('Epoch', i, ' : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
                vectors = sess.run(W1 + b1)

        self.vectors = vectors

    def _train_tf(self, sentences=None):
        words, word2int, int2word = self.build_vocab()

        x_train, y_train = self.get_XY(sentences, word2int)

        self._tf_run(x_train, y_train)

    def _train(self, sentences=None):
        trained = self.model(sentences=sentences, size=self.size,
                             window=self.window, min_count=self.min_count,
                             workers=self.workers, sample=self.sample,
                             negative=self.negative, seed=self.seed)

        trained.save(self.out)

    def train(self, sentences=None):
        if sentences is None:
            if self.sentences is None:
                sentences = self.get_sentences()
            else:
                sentences = self.sentences

        start = time.time()
        if self.tf:
            self._train_tf(sentences)
        else:
            self.train(sentences)
        print("Training {0} took: {1} seconds".format(self.model_name, time.time()-start))
