"""
Multiclass versions for quantifiers based on representing the distributions using CDFs/PDFs
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause

import numpy as np

from sklearn.utils import check_X_y, check_array

from base import UsingClassifiers, WithoutClassifiers
from distribution_matching.utils import golden_section_search, mixture_two_pdfs
from utils import solve_hd, compute_l2_param_train, solve_l1, solve_l2, solve_l2cvx
from scipy.sparse import  csr_matrix, issparse  # JAIME sparse


class DFy(UsingClassifiers):
    """ Generic Multiclass DFy method

        The idea is to represent the mixture of the training distribution and the testing distribution
        (using CDFs/PDFs) of the predictions given by a classifier (y). The difference between both is minimized
        using a distance/loss function. Originally, (González et al. 2013) propose the combination of PDF and
        Hellinger Distance, but also CDF and any other distance/loss function could be used, like L1 or L2. In fact,
        Forman (2005) propose to use CDF's an a measure equivalent to L1.

        The class has two parameters to select:

        - the method used to represent the distributions (CDFs or PDFs)
        - the distance used.

        This class (as every other class based on distribution matching using classifiers) works in two different ways:

        1) Two estimators are used to classify training examples and testing examples in order to
           compute the distribution of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the fit/predict methods. This is useful
           for synthetic/artificial experiments

        The goal in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC and AC/PAC). In the first case, estimators are only trained once and can
        be shared for several quantifiers of this kind

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the distribution of each class individually

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to compute the distribution of the whole testing set

        distribution_function : str, (default='PDF')
            Type of distribution function used. Two types are supported 'CDF' and 'PDF'

        n_bins : int  (default=8)
            Number of bins to compute the CDFs/PDFs

        distance : str, representing the distance function (default='HD')
            It is the name of the distance used to compute the difference between the mixture of the training
            distribution and the testing distribution

        tol : float, (default=1e-05)
            The precision of the solution when search is used to compute the prevalence

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimator_train and estimator_test could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because PDFy quantifiers need to estimate the training distribution
            
        probabilistic_predictions : bool, True
             This means that predictions_train_/predictions_test_ contain probabilistic predictions

        distance : str or a distance function
            A string with the name of the distance function ('HD'/'L1'/'L2') or a distance function

        tol : float
            The precision of the solution when search is used to compute the solution

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit` method whenever the true labels of the training set are needed, instead of y

        distribution_function : str
            Type of distribution function used. Two types are supported 'CDF' and 'PDF'

        n_bins : int
            The number of bins to compute the CDFs/PDFs

        train_distrib_ : ndarray, shape (n_bins * 1, n_classes) binary or (n_bins * n_classes_, n_classes) multiclass
            The CDF/PDF for each class in the training set

        test_distrib_ : ndarray, shape (n_bins * 1, 1) binary quantification or (n_bins * n_classes_, 1) multiclass q
            The CDF/PDF for the testing bag

        G_, C_, b_: variables of different kind for defining the optimization problem
            These variables are precomputed in the `fit` method and are used for solving the optimization problem
            using `quadprog.solve_qp`. See `compute_l2_param_train` function

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        Víctor González-Castro, Rocío Alaiz-Rodríguez, and Enrique Alegre: Class Distribution Estimation based
        on the Hellinger Distance. Information Sciences 218 (2013), 146–164.

        George Forman: Counting positives accurately despite inaccurate classification. In: Proceedings of the 16th
        European conference on machine learning (ECML'05), Porto, (2005) pp 564–575

        Aykut Firat. 2016. Unified Framework for Quantification. arXiv preprint arXiv:1606.00868 (2016).
    """

    def __init__(self, estimator_train=None, estimator_test=None, distribution_function='PDF', n_bins=8, distance='HD',
                 tol=1e-05, verbose=0):
        super(DFy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                  needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # attributes
        self.distribution_function = distribution_function
        self.n_bins = n_bins
        self.distance = distance
        self.tol = tol
        # variables to represent the distributions
        self.train_distrib_ = None
        self.test_distrib_ = None
        # variables for solving the optimization problem
        self.G_ = None
        self.C_ = None
        self.b_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit` method of its superclass.
            After that, the method computes the pdfs for all the classes in the training set

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes)
                Predictions of the examples in the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating training distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        if n_classes == 2:
            n_descriptors = 1  # number of groups of probabilities used to represent the distribution
        else:
            n_descriptors = n_classes

        self.train_distrib_ = np.zeros((self.n_bins * n_descriptors, n_classes))
        # compute pdf
        for n_cls, cls in enumerate(self.classes_):
            for descr in range(n_descriptors):
                self.train_distrib_[descr * self.n_bins:(descr + 1) * self.n_bins, n_cls] = \
                   np.histogram(self.predictions_train_[self.y_ext_ == cls, descr], bins=self.n_bins, range=(0., 1.))[0]
            self.train_distrib_[:, n_cls] = self.train_distrib_[:, n_cls] / (np.sum(self.y_ext_ == cls))

        # compute cdf if necessary
        if self.distribution_function == 'CDF':
            self.train_distrib_ = np.cumsum(self.train_distrib_, axis=0)

        if self.distance == 'L2':
            self.G_, self.C_, self.b_ = compute_l2_param_train(self.train_distrib_, self.classes_)

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            After that, the method computes the PDF for the testing bag.

            Finally, the prevalences are computed using the corresponding function according to the value of
            distance attribute

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a `predict_proba` method)

                If predictions_test is not None they are copied on predictions_test_ and used.
                If predictions_test is None, predictions for the testing examples are computed using the `predict`
                method of estimator_test (it must be an actual estimator)

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are both None

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Estimating testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        if n_classes == 2:
            n_descriptors = 1
        else:
            n_descriptors = n_classes

        self.test_distrib_ = np.zeros((self.n_bins * n_descriptors, 1))
        # compute pdf
        for descr in range(n_descriptors):
            self.test_distrib_[descr * self.n_bins:(descr + 1) * self.n_bins, 0] = \
                np.histogram(self.predictions_test_[:, descr], bins=self.n_bins, range=(0., 1.))[0]

        self.test_distrib_ = self.test_distrib_ / len(self.predictions_test_)

        #  compute cdf if necessary
        if self.distribution_function == 'CDF':
            self.test_distrib_ = np.cumsum(self.test_distrib_, axis=0)

        if self.verbose > 0:
            print('Class %s: Computing prevalences...' % self.__class__.__name__, end='')

        if self.distance == 'HD':
            prevalences = solve_hd(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   n_classes=n_classes)
        elif self.distance == 'L2':
            prevalences = solve_l2(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   G=self.G_, C=self.C_, b=self.b_)
        elif self.distance == 'L2cvx':
            prevalences = solve_l2cvx(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                      n_classes=n_classes)
        elif self.distance == 'L1':
            prevalences = solve_l1(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   n_classes=n_classes)
        else:
            prevalences = golden_section_search(distance_func=self.distance, mixture_func=mixture_two_pdfs,
                                                test_distrib=self.test_distrib_, tol=self.tol,
                                                pos_distrib=self.train_distrib_[:, 1].reshape(-1, 1),
                                                neg_distrib=self.train_distrib_[:, 0].reshape(-1, 1))
        if self.verbose > 0:
            print('done')

        return prevalences


class HDy(DFy):
    """ Multiclass HDy method

        This class is just a wrapper. It just uses all the inherited methods of its superclass (DFy)

        References
        ----------
        Víctor González-Castro, Rocío Alaiz-Rodríguez, and Enrique Alegre: Class Distribution Estimation based
        on the Hellinger Distance. Information Sciences 218 (2013), 146–164.
    """
    def __init__(self, estimator_train=None, estimator_test=None, n_bins=8, tol=1e-05, verbose=0):
        super(HDy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                  distribution_function='PDF', n_bins=n_bins, distance='HD', tol=tol, verbose=verbose)


class DFX(WithoutClassifiers):
    """ Generic Multiclass DFX method

        The idea is to represent the mixture of the training distribution and the testing distribution
        (using CDFs/PDFs) of the features of the input space (X). The difference between both are minimized using a
        distante/loss function. Originally, (González et al. 2013) propose the combination of PDF and
        Hellinger Distance, but also CDF and any other distance/loss function could be used, like L1 or L2.

        The class has two parameters to select:

        - the method used to represent the distributions (CDFs or PDFs)
        - the distance used.

        Parameters
        ----------
        distribution_function : str, (default='PDF')
            Type of distribution function used. Two types are supported 'CDF' and 'PDF'

        n_bins : int
            Number of bins to compute the PDFs

        distance : str, representing the distance function (default='HD')
            It is the name of the distance used to compute the difference between the mixture of the training
            distribution and the testing distribution

        tol : float, (default=1e-05)
            The precision of the solution when search is used to compute the prevalence

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        Attributes
        ----------
        classes_ : ndarray, shape (n_classes, )
            Class labels

        distribution_function : str
            Type of distribution function used. Two types are supported 'CDF' and 'PDF'

        n_bins : int (default=8)
            The number of bins to compute the PDFs

        distance : str or a distance function
            A string with the name of the distance function ('HD'/'L1'/'L2') or a distance function

        tol : float
            The precision of the solution when search is used to compute the solution

        att_ranges_ : ndarray, shape(n_features, 2)
            Min and Max for each feature of the input space

        train_distrib_ : ndarray, shape (n_bins * n_features, n_classes)
            The PDF for each class in the training set

        test_distrib_ : ndarray, shape (n_bins * n_features, 1) multiclass
            The PDF for the testing bag

        verbose : int
            The verbosity level

        References
        ----------
        Víctor González-Castro, Rocío Alaiz-Rodríguez, and Enrique Alegre: Class Distribution Estimation based
        on the Hellinger Distance. Information Sciences 218 (2013), 146–164.

        Aykut Firat. 2016. Unified Framework for Quantification. arXiv preprint arXiv:1606.00868 (2016).
    """
    def __init__(self, distribution_function='PDF', n_bins=8, distance='HD', tol=1e-05, verbose=0):
        # attributes
        self.distribution_function = distribution_function
        self.n_bins = n_bins
        self.distance = distance
        self.tol = tol
        self.verbose = verbose
        # variables to represent the distributions
        self.classes_ = None
        self.att_ranges_ = None
        self.train_distrib_ = None
        self.test_distrib_ = None
        #  variables for solving the optimization problem
        self.G_ = None
        self.C_ = None
        self.b_ = None

    def fit(self, X, y):
        """ This method just computes the PDFs for all the classes in the training set. The values are stored in
            train_dist_

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes
        """
        if self.verbose > 0:
            print('Class %s: Estimating training distribution...' % self.__class__.__name__, end='')

        check_X_y(X, y, accept_sparse=True)

        self.classes_ = np.unique(y)

        self.att_ranges_ = [(a.min(), a.max()) for a in X.T]

        n_classes = len(self.classes_)

        self.train_distrib_ = np.zeros((self.n_bins * X.shape[1], n_classes))
        for n_cls, cls in enumerate(self.classes_):
            # compute pdf
            for att in range(X.shape[1]):
                # JAIME SPARSE
                if issparse(X):
                     X  = X.toarray()
                self.train_distrib_[att * self.n_bins:(att + 1) * self.n_bins, n_cls] = \
                    np.histogram(X[y == cls, att], bins=self.n_bins, range=self.att_ranges_[att])[0]
            self.train_distrib_[:, n_cls] = self.train_distrib_[:, n_cls] / np.sum(y == cls)
            # compute cdf if necessary
            if self.distribution_function == 'CDF':
                for att in range(X.shape[1]):
                    self.train_distrib_[att * self.n_bins:(att + 1) * self.n_bins, n_cls] = \
                        np.cumsum(self.train_distrib_[att * self.n_bins:(att + 1) * self.n_bins, n_cls], axis=0)

        if self.distance == 'L2':
            self.G_, self.C_, self.b_ = compute_l2_param_train(self.train_distrib_, self.classes_)

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X):
        """ Predict the class distribution of a testing bag

            First, the method computes the PDF for the testing bag.

            After that, the prevalences are computed using the corresponding function according to the value of
            distance attribute

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            Returns
            -------
            prevalences: ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        if self.verbose > 0:
            print('Class %s: Estimating testing distribution...' % self.__class__.__name__, end='')

        check_array(X, accept_sparse=True)

        n_classes = len(self.classes_)

        self.test_distrib_ = np.zeros((self.n_bins * X.shape[1], 1))
        # compute pdf
        for att in range(X.shape[1]):
            # JAIME SPARSE
            if issparse(X):
                X = X.toarray()
            self.test_distrib_[att * self.n_bins:(att + 1) * self.n_bins, 0] = \
                np.histogram(X[:, att], bins=self.n_bins, range=self.att_ranges_[att])[0]
        self.test_distrib_ = self.test_distrib_ / len(X)
        # compute cdf if necessary
        if self.distribution_function == 'CDF':
            for att in range(X.shape[1]):
                self.test_distrib_[att * self.n_bins:(att + 1) * self.n_bins, 0] = \
                    np.cumsum(self.test_distrib_[att * self.n_bins:(att + 1) * self.n_bins, 0], axis=0)

        if self.distance == 'HD':
            prevalences = solve_hd(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   n_classes=n_classes)
        elif self.distance == 'L2':
            prevalences = solve_l2(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   G=self.G_, C=self.C_, b=self.b_)
        elif self.distance == 'L2cvx':
            prevalences = solve_l2cvx(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                      n_classes=n_classes)
        elif self.distance == 'L1':
            prevalences = solve_l1(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   n_classes=n_classes)
        else:
            prevalences = golden_section_search(distance_func=self.distance, mixture_func=mixture_two_pdfs,
                                                test_distrib=self.test_distrib_, tol=self.tol,
                                                pos_distrib=self.train_distrib_[:, 1].reshape(-1, 1),
                                                neg_distrib=self.train_distrib_[:, 0].reshape(-1, 1))
        if self.verbose > 0:
            print('done')

        return prevalences


class HDX(DFX):
    """ Multiclass HDX method

        This class is a wrapper. It just uses all the inherited methods of its superclass (DFX)

        References
        ----------
        Víctor González-Castro, Rocío Alaiz-Rodríguez, and Enrique Alegre: Class Distribution Estimation based on
        the Hellinger Distance. Information Sciences 218 (2013), 146–164.
    """
    def __init__(self, n_bins=8, tol=1e-05):
        super(HDX, self).__init__(distribution_function='PDF', n_bins=n_bins, distance='HD', tol=tol)
