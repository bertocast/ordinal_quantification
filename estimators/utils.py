"""
Util functions used for building new estimators
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause

import numpy as np

from copy import copy
from sklearn.preprocessing import LabelBinarizer


class FHLabelBinarizer(LabelBinarizer):
    """ Binarize labels in a Frank and Hall decomposition

        This type of decomposition works as follows. For instance, in a ordinal classification problem with classes
        ranging from 1-star to 5-star, Frank and Hall (FH) decompositon trains 4 binary classifiers:
        1 vs 2-3-4-5, 1-2 vs 3-4-5, 1-2-3 vs 4-5, 1-2-3-4 vs 5 and combines their predictions.

        To train all these binary classifiers, one needs to convert the original ordinal labels to binary labels
        for each of the binary problems of the Frank and Hall decomposition. FHLabelBinarizer makes this process
        easy using the transform method.

         Parameters
         ----------
         neg_label : int (default: 0)
             Value with which negative labels must be encoded.

         pos_label : int (default: 1)
             Value with which positive labels must be encoded.

         sparse_output : boolean (default: False)
             True if the returned array from transform is desired to be in sparse CSR format.
        """
    def __init__(self, neg_label=0, pos_label=1):
        super(FHLabelBinarizer, self).__init__(neg_label=neg_label, pos_label=pos_label, sparse_output=False)

    def transform(self, y):
        """ Transform ordinal labels to the Frank and Hall binary labels

            Parameters
            ----------
            y : array, (n_samples,)
                Class labels for a set of examples

            Returns
            -------
            y_bin_fh : array, (n_samples, n_classes)
                Each column contains the binary labels for the consecutive binary problems of a Frank and Hall
                decomposition from left to right. For instance, in a 4-class problem, each column corresponds to
                the following problems:

                1st column: 1 vs 2-3-4
                2nd column: 1-2 vs 3-4
                3rd column: 1-2-3 vs 4
                4ht column: (not really used)
        """
        y_bin = super().transform(y)
        y_bin_fh = copy(y_bin)
        for i in range(len(self.classes_)):
            y_bin_fh[:, i] = np.sum(y_bin[:, 0:i + 1], axis=1)
        return y_bin_fh

    #def inverse_transform(self, y, threshold=None):
    #    return super().inverse_transform(y)


def check_and_correct_probabilities_asc(probabilities):
    """ This function checks and corrects those probabilities of the binary models of a Frank and Hall estimator
        that are inconsistent. It is used by FrankAndHallMonotoneClassifier.

        To obtain consistent probabilities, we need to ensure that the consecutive probabilities do not decrease.

        Example:

            Classifier 1 vs 2-3-4   Pr({1}) = 0.3
            Classifier 1-2 vs 3-4   Pr({1,2}) = 0.2
            Classifier 1-2-3 vs 4   Pr({1,2,3}) = 0.6

        This is inconsistent. Following (Destercke and Yang, 2014) the method computes the upper (adjusting from
        left to right) and the lower (from right to left) cumulative probabilities. These sets of values are
        monotonically increasing (from left to right) and monotonically decreasing (from right to left),
        respectively. The average value is assigned to each group.

        Example:

            {1}   {1-2}  {1-2-3}

            0.3   0.3    0.6    Upper cumulative probabilities (adjusting from left to right)

            0.2   0.2    0.6    Lower cumulative probabilities (adjusting from right to left)
            ----------------
            0.25  0.25   0.6    Averaged probability

        Parameters
        ----------
        probabilities : array, shape(n_examples, n_classes-1)
            The probabilities of the binary models of a FrankAndHonotone estimator for a complete dataset

        Return
        ------
        probabilities_ok : array, shape(n_examples, n_classes-1)
            The corrected probabilities ensuring that do not decrease (from left to right)

        References
        ----------
        Sébastien Destercke, Gen Yang. Cautious Ordinal Classification by Binary Decomposition.
        Machine Learning and Knowledge Discovery in Databases - European Conference ECML/PKDD,
        Sep 2014, Nancy, France. pp.323 - 337, 2014,
    """
    ascending = np.all(probabilities[:, 1:] >= probabilities[:, :-1], axis=1)
    samples_to_correct = np.nonzero(ascending == False)[0]
    if len(samples_to_correct) == 0:
        return probabilities
    else:
        probabilities_ok = np.copy(probabilities)
    # correct those samples in which their probabilities are incosistent
    for sample in samples_to_correct:
        n = len(probabilities_ok[sample])
        # left to right corrections
        probs1 = np.copy(probabilities_ok[sample])
        for i in range(1, n):
            if probs1[i] < probs1[i - 1]:
                probs1[i] = probs1[i - 1]
        # right to left correction
        probs2 = np.copy(probabilities_ok[sample])
        for i in range(n - 1, 0, -1):
            if probs2[i] < probs2[i - 1]:
                probs2[i - 1] = probs2[i]
        # storing the average of both corrections
        probabilities_ok[sample] = (probs1 + probs2) / 2.0
    return probabilities_ok

