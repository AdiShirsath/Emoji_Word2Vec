import pickle
from argparse import ArgumentParser

from gensim.models import Word2Vec


def preparing_data(args):
    if args.data_path:
        with open(args.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            train_data = [line.strip().split() for line in lines]
        return train_data
    else:
        print(f"Please provide dataset path")


def train_model(args, train_data):
    # train model
    print(f"training word2vec with {args}")
    model = Word2Vec(
        train_data,
        size=args.size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        max_vocab_size=args.vocab_size,
        negative=args.negative,
    )
    print("saving model")
    model.save("word2vec.bin")
    model.wv.save_word2vec_format("word2vec.txt")


def train():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="", help="Path to text dataset which is expected to be .txt file."
    )
    parser.add_argument("--size", type=int, default=300, help="word2vec model's hidden layers size.")
    parser.add_argument("--window", type=int, default=5, help="word2vec model's window")
    parser.add_argument("--min_count", type=int, default=10, help="word2vec model's min_count")
    parser.add_argument("--workers", type=int, default=4, help="word2vec model's window")
    parser.add_argument("--vocab_size", type=int, default=None, help="word2vec vocab size")
    parser.add_argument("--negative", type=int, default=5, help="negative samples")

    # create arg
    args = parser.parse_args()
    train_data = preparing_data(args)
    train_model(args, train_data)


if __name__ == "__main__":
    train()
