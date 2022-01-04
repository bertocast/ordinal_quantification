"""
Multiclass versions of AC and PAC quantifiers
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause

import numpy as np

from sklearn.metrics import confusion_matrix

from base import UsingClassifiers
from utils import compute_l2_param_train, solve_l2, solve_l1, solve_hd, solve_l2cvx


class AC(UsingClassifiers):
    """ Multiclass Adjusted Count method

        This class works in two different ways:

        1) Two estimators are used to classify the examples of the training set and the testing set in order to
           compute the confusion matrix of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the `fit`/`predict methods. This is useful
           for synthetic/artificial experiments

        The idea in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC). In the first case, estimators are only trained once and can be shared
        for several quantifiers of this kind

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict`. It is used to classify the examples of the training
            set and to compute the confusion matrix

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict`. It is used to classify the examples of the testing
            set and to obtain their predictions

        distance : str, representing the distance function (default='HD')
            It is the name of the distance used to compute the difference between the mixture of the training
            distribution and the testing distribution. Only used in multiclass problems.
            Distances supported: 'HD', 'L2' and 'L1'

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimators could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        distance : str
            A string with the name of the distance function ('HD'/'L1'/'L2')

        predictions_train_ : ndarray, shape (n_examples, ) (crisp estimator)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, ) (crisp estimator)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because AC quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, False
             This means that predictions_test_ contains crisp predictions

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit` method whenever the true labels of the training set are needed, instead of y

        cm_ : ndarray, shape (n_classes, n_classes)
            Confusion matrix

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
        George Forman. 2008. Quantifying counts and costs via classification. Data Mining Knowledge Discovery 17,
        2 (2008), 164–206.
    """

    def __init__(self, estimator_train=None, estimator_test=None, distance='HD', verbose=0):
        super(AC, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                 needs_predictions_train=True, probabilistic_predictions=False, verbose=verbose)
        self.distance = distance
        # confusion matrix
        self.cm_ = None
        # variables for solving the optimization problem when n_classes > 2 and distance = 'L2'
        self.G_ = None
        self.C_ = None
        self.b_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (crisp values) if needed. Both operations are
            performed by the `fit` method of its superclass.
            Finally the method computes the confusion matrix of the training set using predictions_train_

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, ) or (n_examples, n_classes)
                Predictions of the examples in the training set. If shape is (n_examples, n_classes) predictions are
                converted to crisp values by `super().fit()

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating confusion matrix for training distribution...' % self.__class__.__name__,
                  end='')

        #  estimating the confusion matrix
        cm = confusion_matrix(self.y_ext_, self.predictions_train_, labels=self.classes_)
        #  normalizing cm by row
        self.cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # binary:  [[1-fpr  fpr]
        #                                                                          [1-tpr  tpr]]

        if len(self.classes_) > 2 and self.distance == 'L2':
            self.G_, self.C_, self.b_ = compute_l2_param_train(self.cm_.T, self.classes_)

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            After that, the prevalences are computed solving a system of linear scalar equations:

                         cm_.T * prevalences = CC(X)

            For binary problems the system is directly solved using the original AC algorithm proposed by Forman

                        p = (p_0 - fpr ) / ( tpr - fpr)

            For multiclass problems, the system may not have a solution. Thus, instead we propose to solve an
            optimization problem of this kind:

                      Min   distance ( cm_.T * prevalences, CC(X) )
                      s.t.  sum(prevalences) = 1
                            prevalecences_i >= 0

            in which distance can be 'HD' (defect value), 'L1' or 'L2'

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a `predict_proba method)

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
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)

        freq = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls, 0] = np.equal(self.predictions_test_, cls).sum()

        prevalences_0 = freq / float(len(self.predictions_test_))

        if n_classes == 2:
            if np.abs((self.cm_[1, 1] - self.cm_[0, 1])) > 0.001:
                p = (prevalences_0[1] - self.cm_[0, 1]) / (self.cm_[1, 1] - self.cm_[0, 1])
                prevalences = [1-p, p]
            else:
                prevalences = prevalences_0

            # clipping the results according to (Forman 2008)
            prevalences = np.clip(prevalences, 0, 1)

            if np.sum(prevalences) > 0:
                prevalences = prevalences / float(np.sum(prevalences))

            prevalences = prevalences.squeeze()
        else:
            if self.distance == 'HD':
                prevalences = solve_hd(train_distrib=self.cm_.T, test_distrib=prevalences_0, n_classes=n_classes)
            elif self.distance == 'L2':
                prevalences = solve_l2(train_distrib=self.cm_.T, test_distrib=prevalences_0,
                                       G=self.G_, C=self.C_, b=self.b_)
            elif self.distance == 'L2cvx':
                prevalences = solve_l2cvx(train_distrib=self.cm_.T, test_distrib=prevalences_0, n_classes=n_classes)
            elif self.distance == 'L1':
                prevalences = solve_l1(train_distrib=self.cm_.T, test_distrib=prevalences_0, n_classes=n_classes)
            else:
                raise ValueError('Class %s": distance function not supported', self.__class__.__name__)

        if self.verbose > 0:
            print('done')

        # print('AC: p %.5f l2 %.5f' %(prevalences[1], l2(self.cm_.T.dot(prevalences), prevalences_0.squeeze())))
        return prevalences


class PAC(UsingClassifiers):
    """ Multiclass Probabilistic Adjusted Count method

        This class works in two different ways:

        1) Two estimators are used to classify the examples of the training set and the testing set in order to
           compute the (probabilistic) confusion matrix of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the `fit`/`predict methods. This is useful
           for synthetic/artificial experiments

        The idea in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC). In the first case, estimators are only trained once and can be shared
        for several quantifiers of this kind

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the confusion matrix

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to obtain the confusion matrix of the testing set

        distance : str, representing the distance function (default='L2')
            It is the name of the distance used to compute the difference between the mixture of the training
            distribution and the testing distribution. Only used in multiclass problems.
            Distances supported: 'HD', 'L2' and 'L1'

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimators could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        distance : str
            A string with the name of the distance function ('HD'/'L1'/'L2')

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilistic estimator)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilistic estimator)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because PAC quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
             This means that predictions_test_ contains probabilistic predictions

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit` method whenever the true labels of the training set are needed, instead of y


        cm_ : ndarray, shape (n_classes, n_classes)
            Confusion matrix

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
        Antonio Bella, Cèsar Ferri, José Hernández-Orallo, and María José Ramírez-Quintana. 2010. Quantification
        via probability estimators. In Proceedings of the IEEE International Conference on Data Mining (ICDM’10).
        IEEE, 737–742.
    """

    def __init__(self, estimator_test=None, estimator_train=None, distance='L2', verbose=0):
        super(PAC, self).__init__(estimator_test=estimator_test, estimator_train=estimator_train,
                                  needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        self.distance = distance
        # confusion matrix with average probabilities
        self.cm_ = None
        # variables for solving the optimization problem when n_classes > 2 and distance = 'L2'
        self.G_ = None
        self.C_ = None
        self.b_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit method of its superclass.
            Finally the method computes the (probabilistic) confusion matrix using predictions_train

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes)
                Predictions of the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating average probabilities for training distribution...'
                  % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        # estimating the confusion matrix
        # average probabilty distribution for each class
        self.cm_ = np.zeros((n_classes, n_classes))
        for n_cls, cls in enumerate(self.classes_):
            self.cm_[n_cls] = np.mean(self.predictions_train_[self.y_ext_ == cls], axis=0)

        if len(self.classes_) > 2 and self.distance == 'L2':
            self.G_, self.C_, self.b_ = compute_l2_param_train(self.cm_.T, self.classes_)

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict() method.

            After that, the prevalences are computed solving a system of linear scalar equations:

                         cm_.T * prevalences = PCC(X)

            For binary problems the system is directly solved using the original PAC algorithm proposed by Bella et al.

                        p = (p_0 - PA(negatives) ) / ( PA(positives) - PA(negatives) )

            in which PA stands for probability average.

            For multiclass problems, the system may not have a solution. Thus, instead we propose to solve an
            optimization problem of this kind:

                      Min   distance ( cm_.T * prevalences, PCC(X) )
                      s.t.  sum(prevalences) = 1
                            prevalecences_i >= 0

            in which distance can be 'HD', 'L1' or 'L2' (defect value)

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
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        prevalences_0 = np.mean(self.predictions_test_, axis=0)

        if n_classes == 2:
            if np.abs(self.cm_[1, 1] - self.cm_[0, 1]) > 0.001:
                p = (prevalences_0[1] - self.cm_[0, 1]) / (self.cm_[1, 1] - self.cm_[0, 1])
                prevalences = [1 - p, p]
            else:
                prevalences = prevalences_0
            # prevalences = np.linalg.solve(self.cm_.T, prevalences_0)

            # clipping the results according to (Forman 2008)
            prevalences = np.clip(prevalences, 0, 1)

            if np.sum(prevalences) > 0:
                prevalences = prevalences / float(np.sum(prevalences))

            prevalences = prevalences.squeeze()
        else:
            if self.distance == 'HD':
                prevalences = solve_hd(train_distrib=self.cm_.T, test_distrib=prevalences_0, n_classes=n_classes)
            elif self.distance == 'L2':
                prevalences = solve_l2(train_distrib=self.cm_.T, test_distrib=prevalences_0,
                                       G=self.G_, C=self.C_, b=self.b_)
            elif self.distance == 'L2cvx':
                prevalences = solve_l2cvx(train_distrib=self.cm_.T, test_distrib=prevalences_0, n_classes=n_classes)
            elif self.distance == 'L1':
                prevalences = solve_l1(train_distrib=self.cm_.T, test_distrib=prevalences_0, n_classes=n_classes)
            else:
                raise ValueError('Class %s": distance function not supported', self.__class__.__name__)

        if self.verbose > 0:
            print('done')

        # print('PAC: p %.5f l2 %.5f' %(prevalences[1], l2(self.cm_.T.dot(prevalences), prevalences_0.squeeze())))
        return prevalences


class DeBias(UsingClassifiers):
    """ Binary quantifier based on De-Bias estimate proposed by Friedman

        prevalence (positives) = prior(positives)  + ( prevalence_PCC - prior(positives) ) / Vt

        where

        Vt =[ 1/|T| sum_{x in D} (P(h(x)==+1|x) - prior(positives) )^2 ] / (prior(positives) * prior(negatives))

        This class works in two different ways:

        1) An estimator is used to classify the examples of the testing bag (the estimator can be already trained)

        2) You can directly provide the predictions for the examples in the predict method. This is useful
           for synthetic/artificial experiments

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the confusion matrix

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to obtain the confusion matrix of the testing set

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimators could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set.

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilistic estimator)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilistic estimator)
            Predictions of the examples in the testing bag

        probabilistic_predictions : bool, True
             This means that predictions_train_/predictions_test_ contain probabilistic predictions

        needs_predictions_train : bool, True
            It is True because DeBias quantifiers need to estimate the training distribution

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(n_examples, )
            True labels of the training set

        train_prevs_ : ndarray, shape (n_classes, )
            Prevalence of each class in the training set

        Vt_ : float
           The value of equation
                Vt =[ 1/|T| sum_{x in D} (P(h(x)==+1|x) - train_prevs_[1])^2 ] / (train_prevs_[1] * train_prevs_[0])
           applied over the training examples D

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        Jerome Friedman. Class counts in future unlabeled samples. Presentation at MIT CSAIL Big Data Event, 2014.
    """

    def __init__(self, estimator_test=None, estimator_train=None, verbose=0):
        super(DeBias, self).__init__(estimator_test=estimator_test, estimator_train=estimator_train,
                                     needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # priors
        self.train_prevs_ = None
        # Vt value
        self.Vt_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit method of its superclass.

            Finally the method computes the value of Vt

            Vt =[ 1/|T| sum_{x in D} (P(h(x)==+1|x) - prior(positives) )^2 ] / (prior(positives) * prior(negatives))

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes)
                Predictions of the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
            AttributeError
                When the number of classes > 2
        """
        if len(np.unique(y)) != 2:
            raise AttributeError("DB is a binary method, multiclass quantification is not supported")

        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating Vt for training distribution...' % self.__class__.__name__, end='')

        self.train_prevs_ = np.unique(y, return_counts=True)[1] / len(X)

        At = np.mean((self.predictions_train_[:, 1] - self.train_prevs_[1]) ** 2)
        self.Vt_ = At / (self.train_prevs_[1] * self.train_prevs_[0])

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            The prevalence for the positive class is

            prevalence (positives) = prior(positives)  + ( prevalence_PCC - prior(positives) ) / Vt

            Parameters
            ----------
            X : (sparse) array-like, shape (n_examples, n_features)
                Data

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a predict_proba method)

                If predictions_test is not None they are copied on predictions_test_ and used.
                If predictions_test is None, predictions for the testing examples are computed using the `predict`
                method of estimator_test (it must be an actual estimator)

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are both None

            Returns
            -------
            prevalences : An ndarray, shape(n_classes, ) with the prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        p = self.train_prevs_[1] + (np.mean(self.predictions_test_, axis=0)[1] - self.train_prevs_[1]) / self.Vt_
        prevalences = np.array([1 - p, p])

        prevalences = np.clip(prevalences, 0, 1)

        if np.sum(prevalences) > 0:
            prevalences = prevalences / float(np.sum(prevalences))

        if self.verbose > 0:
            print('done')

        return prevalences
