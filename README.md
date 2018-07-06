# wiki2vec
<p>Training W2V and FastText on Wikipedia dumps.</p>
<p>Made for job-submissions on computing clusters</p>
<p><img src="https://upload.wikimedia.org/wikipedia/en/thumb/8/80/Wikipedia-logo-v2.svg/1122px-Wikipedia-logo-v2.svg.png" width=250></p>

## Usage

```bash
wget -P data/  https://dumps.wikimedia.org/enwiki/latest/<wiki_dump_name>
```

<p>
e.g. <wiki_dump_name> enwiki-latest-pages-articles1.xml-p10p30302.bz2
  </br>
Dumps should be in <bold>bz2</bold> format, as they are processed using: </p>

```python
from gensim.corpora import WikiCorpus
wiki = WikiCorpus(self.inp, lemmatize=False, dictionary={})
```

```bash
python3.5 train.py <input (bz2 wiki dump) path> <output (trained model) path>
```
### CLI Arguments:

- `i`: input (bz2 wiki dump) path *required
- `o`: output (trained model) path *required
- `--ft`: use FastText (bool: default True)
- `--sg`: use SkipGram (bool: default True)
- `--window`: context window size (int: default 300)
- `--min_word_count`: min word count (int: default 3)
- `--workers`: utilized cores count (int: default all)
- `--sample`: subsampling ratio (float: default 0.001)
- `--negative`: negative samples count (int: default 5)
- `--seed`: random seed (int: default 48)


## Dependancies
Tested on Python 3.5
- <a href="https://radimrehurek.com/gensim/"> gensim </a>
- <a href="https://www.nltk.org/"> NLTK </a>
- <a href="https://pythonhosted.org/joblib/"> joblib </a>

## Useful Tutorials
### Siraj Raval Word2Vec Live tutorial

- https://youtu.be/pY9EwZ02sXU
- https://github.com/llSourcell/word_vectors_game_of_thrones-LIVE

### Chris McCormick Skip-Gram for theory
- http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model
