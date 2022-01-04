"""
Util functions used for building quantifiers based on decomposition
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause

import numpy as np
# from sklearn.utils import check_consistent_length, safe_indexing


def check_and_correct_prevalences_asc(prevalences):
    """ This function checks and corrects the prevalences of a quantifier based on the Frank and Hall decomposition
        that are inconsistent. It is used by FrankAndHallQuantifier.

        To obtain consistent prevalences, we need to ensure that the consecutive probabilities do not decrease.

        Example:

            Quantifier 1 vs 2-3-4   Prevalence({1}) = 0.3
            Quantifier 1-2 vs 3-4   Prevalence({1,2}) = 0.2
            Quantifier 1-2-3 vs 4   Prevalence({1,2,3}) = 0.6

        This is inconsistent. Following (Destercke, Yang, 2014) the method computes the upper (adjusting from
        left to right) and the lower (from right to left) cumulative prevalences. These sets of values are
        monotonically increasing (from left to right) and monotonically decreasing (from right to left),
        respectively. The average value is assigned to each group

        Example:

            {1}   {1-2}  {1-2-3}

            0.3   0.3    0.6    Upper cumulative prevalences (adjusting from left to right)

            0.2   0.2    0.6    Lower cumulative prevalences (adjusting from right to left)
            ----------------
            0.25  0.25   0.6    Averaged prevalences

        Parameters
        ----------
        prevalences : array, shape(n_classes-1, )
            The prevalences of the binary quantifiers of a FrankAndHallQuantifier for a single dataset

        Return
        ------
        prevalences_ok : array, shape(n_classes-1)
            The corrected prevalences ensuring that do not decrease (from left to right)

        References
        ----------
        Sébastien Destercke, Gen Yang. Cautious Ordinal Classification by Binary Decomposition.
        Machine Learning and Knowledge Discovery in Databases - European Conference ECML/PKDD,
        Sep 2014, Nancy, France. pp.323 - 337, 2014,
    """
    ascending = np.all(prevalences[1:] >= prevalences[:-1])
    if ascending:
        return prevalences
    n = len(prevalences)
    # left to right corrections
    prevs1 = np.copy(prevalences)
    for i in range(1, n):
        if prevs1[i] < prevs1[i - 1]:
            prevs1[i] = prevs1[i - 1]
    # right to left correction
    prevs2 = np.copy(prevalences)
    for i in range(n - 1, 0, -1):
        if prevs2[i] < prevs2[i - 1]:
            prevs2[i - 1] = prevs2[i]
    # returning the average of both corrections
    return (prevs1 + prevs2) / 2.0

# def ovo_binarizer(X, y, i, j):
#     """Fit a single binary estimator (one-vs-one)."""
#     cond = np.logical_or(y == i, y == j)
#     #y = y[cond]  #Modifica la y !!!!!
#     y_bin = y[cond]
#     y_binary = np.empty(y_bin.shape, np.int)
#     y_binary[y_bin == i] = 0
#     y_binary[y_bin == j] = 1
#     indcond = np.arange(X.shape[0])[cond]
#     X_subset = safe_indexing(X, indices=indcond)
#     return X_subset, y_binary
