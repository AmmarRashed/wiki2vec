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
python3.5 train.py <dump_path> <trained_model_storing_path> <model_name>
```
#### Supported model names are:
<p>"w2v_sg": Word2Vec with Skip-Gram </br>
"w2v_cbow": Word2Vec with CBOW</br>
"ft_sg": FastText with Skip-Gram</br>
"ft_cbow": FastText with CBOW
</p>

<p> The script should ask for the following parameters:</p>


```python
Window size (default 300)
min_word_count (default 3)
Workers (default all)
Subsampling ratio (default 1e-3)
Negative samples (default 5)
random_seed (default 48)
```


## Dependancies
Tested on Python 3.5
<p><a href="https://radimrehurek.com/gensim/"> gensim </a></p>
<p><a href="https://www.nltk.org/"> NLTK </a></p>

## Useful Tutorials
### Siraj Raval Word2Vec Live tutorial

<p> https://youtu.be/pY9EwZ02sXU </p>
<p> https://github.com/llSourcell/word_vectors_game_of_thrones-LIVE </p>

### Chris McCormick Skip-Gram for theory
<p> http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/</p>
