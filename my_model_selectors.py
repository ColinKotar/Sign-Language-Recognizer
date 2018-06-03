import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value
        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score
    """

    def bic_score(self, n):
        """
        http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
        BIC score = -2 * logL + p * logN
            where, p = n**2 + 2 * d * n - 1
                   logL = model score of inputs and lengths
                   logN = log of length of inputs
        """
        model = self.base_model(n) # base model

        # all variables
        logL = model.score(self.X, self.lengths)
        d = model.n_features
        p = n**2 + 2 * d * n - 1
        logN = np.log(len(self.X))

        # BIC score using formula
        bic_score = -2.0 * logL + p * logN

        return bic_score, model


    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_score, best_model = float("inf"), None

        try: # try for best model with best BIC score
            for n in range(self.min_n_components, self.max_n_components + 1):
                bic_score, model = self.bic_score(n)
                if bic_score < best_score:
                    best_score, best_model = bic_score, model
            return best_model
        except: # return base model
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    '''

    def dic_score(self, n):
        """
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
        https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
        DIC score = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        """
        model = self.base_model(n)
        scores = []

        # sum using scores
        for word, (X, lengths) in self.hwords.items():
            if word != self.this_word:
                scores.append(model.score(X, lengths))

        # DIC score using formula
        dic_score = model.score(self.X, self.lengths) - np.mean(scores)

        return dic_score, model


    def select(self):
        """ select the best model for self.this_word based on
        DIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_score, best_model = float("inf"), None

        try: # try for best model with best DIC score
            for n in range(self.min_n_components, self.max_n_components + 1):
                dic_score, model = self.dic_score(n)
                if dic_score < best_score:
                    best_score, best_model = dic_score, model
            return best_model
        except: # return base model
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def cv_score(self, n):
        """
        Calculate the average log likelihood of cross-validation folds using the KFold class
        :return: tuple of the mean likelihood and the model with the respective score
        """
        scores = []
        split_method = KFold(n_splits=2)

        for train, test in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(train, self.sequences)

            # start with base model
            model = self.base_model(n)
            X, l = combine_sequences(test, self.sequences)

            # append scores
            scores.append(model.score(X, l))

        # CV score using formula
        cv_score = np.mean(scores)

        return cv_score, model


    def select(self):
        """ select best model based on average log Likelihood of cross-validation folds
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_score, best_model = float("inf"), None

        try: # try for best model with best CV score
            for n in range(self.min_n_components, self.max_n_components + 1):
                cv_score, model = self.cv_score(n)
                if cv_score < best_score:
                    best_score, best_model = cv_score, model
            return best_model
        except: # return base model
            return self.base_model(self.n_constant)
