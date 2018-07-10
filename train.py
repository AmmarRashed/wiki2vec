import argparse, multiprocessing
from WordEmbeddings import WordEmbeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("i",help="input (bz2 wiki dump) path")
    parser.add_argument("o",help="output (trained model) path")
    parser.add_argument("--ft", help="use FastText (bool: default True)", default=True, type=bool)
    parser.add_argument("--sg", help="use SkipGram (bool: default True)", default=True, type=bool)
    parser.add_argument("--window", help="context window size (int: default 10)", default=10, type=int)
    parser.add_argument("--size", help="Dimensionality of the word vectors (int: default 300)", default=300, type=int)
    parser.add_argument("--min_word_count", help="min word count (int: default 3)", default=3, type=int)
    parser.add_argument("--workers", help="utilized cores count (int: default all)",
                        default=multiprocessing.cpu_count(), type=int)
    parser.add_argument("--sample", help="sub-sampling ratio (float: default 0.001)", default=0.001, type=float)
    parser.add_argument("--negative", help="negative samples count (int: default 5)", default=5, type=int)
    parser.add_argument("--seed", help="random seed (int: default 48)", default=48, type=int)
    parser.add_argument("--tf", help="Use TensorFlow (bool: default False)", default=False, type=bool)
    parser.add_argument("--gpu", help="Use gpu (bool: default False, --tf must be True)", default=False, type=bool)

    args = parser.parse_args()

    inp = args.i
    out = args.o
    if args.ft:
        model = "ft_"
    else:
        model = "w2v_"

    if args.sg:
        model += "sg"
    else:
        model += "cbow"

    window = args.window
    size = args.size
    min_word_count = args.min_word_count
    workers = args.workers
    sample = args.sample
    negative = args.negative
    seed = args.seed
    tf = args.tf
    gpu = args.gpu

    we = WordEmbeddings(inp, out,
                        model_name=model,
                        window=window,
                        size=size,
                        min_count=min_word_count,
                        workers=workers,
                        sample=sample,
                        negative=negative,
                        seed=seed,
                        tf=tf,
                        gpu=gpu)
    we.train()
