import sys, multiprocessing
from WordEmbeddings import WordEmbeddings

models = {"{0}_{1}".format(model, variation) for model in ["w2v", "ft"] for variation in {"sg","cbow"}}

if __name__ == '__main__':
    program = sys.argv[0]
    inp = sys.argv[1]
    out = sys.argv[2]
    try:
        model = sys.argv[3]
        if model not in models:
            print("Model not specified, using FastText Skip-Gram by default")
            raise IndexError
    except IndexError:
        print("Model not specified, using FastText Skip-Gram by default")

        model = "ft_sg"
    try:
        use_defaults = bool(sys.argv[4])
    except IndexError:
        use_defaults = False

    if use_defaults:
        window = 300
        min_word_count = 3
        workers = multiprocessing.cpu_count()
        subsampling = 0.001
        negative = 5
        seed = 48
    else:
        window = int(input("Window size (default 300): ") or 300)
        min_word_count = int(input("min_word_count (default 3): ") or 3)
        workers = int(input("Workers (default all): ") or multiprocessing.cpu_count())
        subsampling = float(input("Subsampling ratio (default 1e-3): ") or 0.001)
        negative = int(input("Negative samples (default 5): ") or 5)
        seed = int(input("random_seed (default 48): ") or 48)

    we = WordEmbeddings(inp, out,
                        model_name=model,
                        window=window,
                        min_count=min_word_count,
                        workers=workers,
                        sample=subsampling,
                        negative=negative,
                        seed=seed)
    we.train()
