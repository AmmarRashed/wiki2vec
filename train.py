import argparse, multiprocessing
from WordEmbeddings import WordEmbeddings

models = {"{0}_{1}".format(model, variation) for model in ["w2v", "ft"] for variation in {"sg","cbow"}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("i",help="path to the input bz2 wiki dump")
    parser.add_argument("o",help="path to store the trained model in")
    parser.add_argument("--ft", help="Use FastText (default 1)", default=1, type=int)
    parser.add_argument("--sg", help="Use SkipGram (default 1)", default=1, type=int)
    parser.add_argument("--window", help="context window size (default 300)", default=300, type=int)
    parser.add_argument("--min_word_count", help="min word count (default 3)", default=3, type=int)
    parser.add_argument("--workers", help="utilized cores count (default all)", default=multiprocessing.cpu_count(), type=int)
    parser.add_argument("--sample", help="subsampling ratio (default 0.001)", default=0.001, type=float)
    parser.add_argument("--negative", help="negative samples count (default 5)", default=5, type=int)
    parser.add_argument("--seed", help="random seed (default 48)", default=48, type=int)

    args = parser.parse_args()

    inp = args.i
    out = args.o
    if int(args.ft):
        model = "ft_"
    else:
        model = "w2v_"

    if int(args.sg):
        model += "sg"
    else:
        model += "cbow"



    window = args.window
    min_word_count = args.min_word_count
    workers = args.workers
    subsampling = args.sample
    negative = args.negative
    seed = args.seed

    we = WordEmbeddings(inp, out,
                        model_name=model,
                        window=window,
                        min_count=min_word_count,
                        workers=workers,
                        sample=subsampling,
                        negative=negative,
                        seed=seed)
    we.train()
