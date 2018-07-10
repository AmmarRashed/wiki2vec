# wiki2vec
<p>Training W2V and FastText on Wikipedia dumps.</p>
<p>Made for job-submissions on computing clusters</p>
<p><img src="https://upload.wikimedia.org/wikipedia/en/thumb/8/80/Wikipedia-logo-v2.svg/1122px-Wikipedia-logo-v2.svg.png" width=250></p>

## Usage

```bash
wget -P data/  https://dumps.wikimedia.org/enwiki/latest/<wiki_dump_name>
```

Dumps should be in _bz2_ format, as they are processed using

```python
from gensim.corpora import WikiCorpus
wiki = WikiCorpus(self.inp, lemmatize=False, dictionary={})
```

```bash
python3.5 train.py <input (bz2 wiki dump) path> <output (trained model) path>
```
### CLI Arguments:

- `i`: input (bz2 wiki dump) path *required* e.g. enwiki-latest-pages-articles1.xml-p10p30302.bz2
- `o`: output (trained model) path *required* 
- `--ft`: use FastText (bool: default True)
- `--sg`: use SkipGram (bool: default True)
- `--window`: context window size (int: default 300)
- `--size`:Dimensionality of the word vectors (int: default 300)
- `--min_word_count`: min word count (int: default 3)
- `--workers`: utilized cores count (int: default all)
- `--sample`: sub-sampling ratio (float: default 0.001)
- `--negative`: negative samples count (int: default 5)
- `--seed`: random seed (int: default 48)
- `--tf`: Use TensorFlow (bool: default False)
- `--gpu`: Use gpu (bool: default False, --tf must be True)


## Dependancies

**Tested on Python 3.5
- <a href="https://radimrehurek.com/gensim/"> gensim </a>
- <a href="https://www.nltk.org/"> NLTK </a>
- <a href="https://pythonhosted.org/joblib/"> joblib </a>
- <a href="https://www.tensorflow.org/"> TensorFlow  </a>(if using `--tf True`)

## Useful Tutorials

- <a href="https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac">Learn Word2Vec by implementing it in tensorflow</a>
- <a href="https://youtu.be/pY9EwZ02sXU">Siraj Raval Word2Vec Live tutorial</a>
- <a href="http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model"> Chris McCormick Skip-Gram for theory </a>

