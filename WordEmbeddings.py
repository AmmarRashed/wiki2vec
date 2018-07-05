import ntpath,os
import multiprocessing


import re
import time

from gensim.corpora import WikiCorpus
from joblib import Parallel, delayed  # for parallel computation

import nltk
from gensim.models.phrases import Phraser, Phrases


class WordEmbeddings(object):
    def __init__(self, inp, out, model_name="ft_sg",
                 window=300, min_count=3,
                 workers=multiprocessing.cpu_count(),
                 sample=0.001,
                 negative=5,
                 seed=48):

        self.model_name = model_name

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
        self.min_count = min_count
        self.workers = workers
        self.sample = sample
        self.negative = negative
        self.seed = seed

    @staticmethod
    def sentence_to_wordlist(raw: str):
        return re.sub("[^a-zA-Z]", " ", raw).split()

    def clean_wiki(self, wiki, output_path="data"):
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

    def get_sentences(self):
        raw_sentences = WikiCorpus(self.inp, lemmatize=False, dictionary={})

        tokenized_sentences = Parallel(n_jobs=self.workers)(
            delayed(self.sentence_to_wordlist)(
                raw_sentence) for raw_sentence in raw_sentences)

        phrases = Phrases(tokenized_sentences)
        bigram = Phraser(phrases)

        sentences = list(bigram[tokenized_sentences])

        return sentences

    def train(self, sentences=None):
        start = time.time()

        if sentences is None:
            sentences = self.get_sentences()

        trained = self.model(sentences=sentences, window=self.window, min_count=self.min_count, workers=self.workers,
                             sample=self.sample, negative=self.negative, seed=self.seed)

        trained.save(self.out)
        print("Training {0} took: {1} seconds".format(self.model_name, time.time()-start))
