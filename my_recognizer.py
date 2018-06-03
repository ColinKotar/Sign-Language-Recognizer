import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for i in range(test_set.num_items):

        best_score = float("-inf")
        best_word = None
        word_scores = {}
        X, lengths = test_set.get_item_Xlengths(i)

        for w, m in models.items():
            try:
                word_scores[w] = m.score(X, lengths)
            except:
                word_scores[w] = float("-inf")

            # update best_score & best_word
            if word_scores[w] > best_score:
                best_score, best_word = word_scores[w], w

        probabilities.append(word_scores)
        guesses.append(best_word)

    return probabilities, guesses
