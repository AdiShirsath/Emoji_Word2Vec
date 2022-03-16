from gensim.models import Word2Vec
from src.utils import is_emoji


class Prediction:
    def __init__(self, model_path: str):
        self.model = Word2Vec.load(model_path)

    def getPrediction(self, pos, neg="", emoji_only=True, topn: int = 500):
        pos = pos.strip().split()
        neg = neg.strip().split()

        outputs = self.model.most_similar(pos, neg, topn=topn)

        if emoji_only:
            outputs = [x for x in outputs if is_emoji(x)]

        return outputs

    def get_similarity(self, w1, w2):
        """
    Get similarity between two words
    """
        return self.model.wv.similarity(w1, w2)

    def get_vector_embedding(self, word: str):
        """
    returns word embedding of input word
    """
        return self.model.wv[word]
