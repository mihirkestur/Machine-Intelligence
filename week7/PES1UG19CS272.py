import numpy as np
import math
from pandas.core.indexing import need_slice
from sklearn.tree import DecisionTreeClassifier

"""
Use DecisionTreeClassifier to represent a stump.
------------------------------------------------
DecisionTreeClassifier Params:
    critereon -> entropy
    max_depth -> 1
    max_leaf_nodes -> 2
Use the same parameters
"""
# REFER THE INSTRUCTION PDF FOR THE FORMULA TO BE USED 

class AdaBoost:

    """
    AdaBoost Model Class
    Args:
        n_stumps: Number of stumps (int.)
    """

    def __init__(self, n_stumps=20):

        self.n_stumps = n_stumps
        self.stumps = []

    def fit(self, X, y):
        """
        Fitting the adaboost model
        Args:
            X: M x D Matrix(M data points with D attributes each)(numpy float)
            y: M Vector(Class target for all the data points as int.)
        Returns:
            the object itself
        """
        self.alphas = []

        sample_weights = np.ones_like(y) / len(y)
        for _ in range(self.n_stumps):

            st = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2)
            st.fit(X, y, sample_weights)
            y_pred = st.predict(X)

            self.stumps.append(st)

            error = self.stump_error(y, y_pred, sample_weights=sample_weights)
            alpha = self.compute_alpha(error)
            self.alphas.append(alpha)
            sample_weights = self.update_weights(
                y, y_pred, sample_weights, alpha)

        return self

    def stump_error(self, y, y_pred, sample_weights):
        """
        Calculating the stump error
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
        Returns:
            The error in the stump(float.)
        """
        # For every instance compare prediction and actual value
        error = 0
        for i in range(len(y_pred)): 
            if(y_pred[i] != y[i]): error += sample_weights[i]
        return error

    def compute_alpha(self, error):
        """
        Computing alpha
        The weight the stump has in the final prediction
        Use eps = 1e-9 for numerical stabilty.
        Args:
            error:The stump error(float.)
        Returns:
            The alpha value(float.)
        """
        # Apply formula for alpha calculation
        eps = 1e-9
        if(error == 0): error = eps
        alpha = (1/2)*(np.log((1-error)/error))
        return alpha

    def update_weights(self, y, y_pred, sample_weights, alpha):
        """
        Updating Weights of the samples based on error of current stump
        The weight returned is normalized
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
            alpha: The stump weight(float.)
        Returns:
            new_sample_weights:  M Vector(new Weight of each sample float.)
        """
        error = self.stump_error(y, y_pred, sample_weights)
        new_sample_weights = sample_weights
        if(error != 0):
            N = 2 * (math.sqrt(error * (1 - error)))
            for i in range(len(y_pred)):
                if(y_pred[i] == y[i]): new_sample_weights[i] = (sample_weights[i]/N) * (math.exp(-alpha))
                else: new_sample_weights[i] = (sample_weights[i]/N) * (math.exp(alpha))
        return new_sample_weights

    def predict(self, X):
        """
        Predicting using AdaBoost model with all the decision stumps.
        Decison stump predictions are weighted.
        Args:
            X: N x D Matrix(N data points with D attributes each)(numpy float)
        Returns:
            pred: N Vector(Class target predicted for all the inputs as int.)
        """
        sign_funcs = []
        # Calculate sum of alphas
        #alpha_sum = sum(self.alphas)
        max_a = max(self.alphas)
        ind = 0
        for i in range(len(self.alphas)):
            if(self.alphas[i] == max_a):
                ind = i
                break
        # Obtain predictions for each classifier
        stump_predictions = [self.stumps[i].predict(X) for i in range(self.n_stumps)]
        # note total samples
        n_samples = len(stump_predictions[0])
        # For each sample calculate sign function and divide by alpha sum to obtain multi class prediction
        for i in range(n_samples):
            # sign_func_sum = 0
            # for j in range(self.n_stumps):
            #     sign_func_sum += self.alphas[j] * stump_predictions[j][i]
            # sign_funcs.append(sign_func_sum)
            sign_funcs.append(stump_predictions[ind][i])
        return sign_funcs

    def evaluate(self, X, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix
            y: True target of test data
        Returns:
            accuracy : (float.)
        """
        pred = self.predict(X)
        # find correct predictions
        correct = (pred == y)

        accuracy = np.mean(correct) * 100  # accuracy calculation
        return accuracy