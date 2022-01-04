"""
Util functions for distribution matching methods
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause

import numpy as np
import math
import quadprog
from utils import is_pd, nearest_pd


############
# Golden section Search
############
def golden_section_search(distance_func, mixture_func, test_distrib, tol, **kwargs):
    """ Golden section search

        Used by PDF and quantiles classes. Only useful for binary quantification
        Given a function `distance_func` with a single local minumum in the interval [0,1], `golden_section_search`
        returns the prevalence that minimizes the differente between the mixture training distribution and
        the testing distribution according to `distance_func`

        Parameters
        ----------
        distance_func : function
            This is the loss function minimized during the search

        mixture_func : function
            The function used to generated the training mixture distribution given a value for the prevalence

        test_distrib : array
            The distribution of the positive class. The exact shape depends on the representation (pdfs, quantiles...)

        tol : float
            The precision of the solution

        kwargs : keyword arguments
            Here we pass the set of arguments needed by mixture functions: mixture_two_pdfs (for pdf-based classes) and
            compute quantiles (for quantiles-based classes). See the help of this two functions

        Returns
        -------
        prevalences : array, shape(2,)
           The predicted prevalence for the negative and the positive class
    """
    #  uncomment the following line for checking whether the distance function is V-shape or not
    assert is_V_shape(distance_func, mixture_func, test_distrib, 0.01, False, **kwargs)

    # some constants
    invphi = (math.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1/phi^2

    a = 0
    b = 1

    h = b - a

    # required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h

    train_mixture_distrib = mixture_func(prevalence=c, **kwargs)
    fc = distance_func(train_mixture_distrib, test_distrib)
    train_mixture_distrib = mixture_func(prevalence=d, **kwargs)
    fd = distance_func(train_mixture_distrib, test_distrib)

    for k in range(n - 1):
        if fc < fd:
            b = d
            d = c
            fd = fc
            h = invphi * h
            c = a + invphi2 * h
            train_mixture_distrib = mixture_func(prevalence=c, **kwargs)
            fc = distance_func(train_mixture_distrib, test_distrib)

        else:
            a = c
            c = d
            fc = fd
            h = invphi * h
            d = a + invphi * h
            train_mixture_distrib = mixture_func(prevalence=d, **kwargs)
            fd = distance_func(train_mixture_distrib, test_distrib)

    if fc < fd:
        return np.array([1 - (a + d) / 2, (a + d) / 2])
    else:
        return np.array([1 - (c + b) / 2, (c + b) / 2])


def mixture_two_pdfs(prevalence=None, pos_distrib=None, neg_distrib=None):
    """ Mix two pdfs given a value por the prevalence of the positive class

        Parameters
        ----------
        prevalence : float,
           The prevalence for the positive class

        pos_distrib : array, shape(n_bins,)
            The distribution of the positive class. The exact shape depends on the representation (pdfs, quantiles...)

        neg_distrib : array, shape(n_bins,)
            The distribution of the negative class. The exact shape depends on the representation (pdfs, quantiles...)

        Returns
        -------
        mixture : array, same shape of positives and negatives
           The pdf mixture of positives and negatives
    """
    mixture = pos_distrib * prevalence + neg_distrib * (1 - prevalence)
    return mixture


def compute_quantiles(prevalence=None, probabilities=None, n_quantiles=None, y=None, classes=None):
    """ Compute quantiles

        Used by quantiles-based classes. It computes the quantiles both for the testing distribution (in this case
        the value of the prevalence is ignored), and for the weighted mixture of positives and negatives (this depends
        on the value of the prevalence parameter)

        Parameters
        ----------
        prevalence : float or None
            The value of the prevalence of the positive class to compute the mixture of the positives and the negatives.
            To compute the quantiles of the testing set this parameter must be None

        probabilities : ndarray, shape (nexamples, 1)
            The ordered probabilities for all examples. Notice that in the case of computing the mixture of the
            positives and the negatives, this array contains the probability for all the examples of the training set

        n_quantiles : int
            Number of quantiles. This parameter is used with Quantiles-based algorithms.

        y : array, labels
            This parameter is used with Quantiles-based algorithms. They need the true label of each example

        classes: ndarray, shape (n_classes, )
            Class labels. Used by Quantiles-based algorithms

        Returns
        -------
        quantiles : array, shape(n_quantiles,)
           The value of the quantiles given the probabilities (and the value of the prevalence if we are computing the
           quantiles of the training mixture distribution)
    """

    # by default (test set) the weights are all equal
    p_weight = np.ones(len(probabilities))
    if prevalence is not None:
        # train set
        n = 1 - prevalence
        n_negatives = np.sum(y == classes[0])
        n_positives = np.sum(y == classes[1])
        p_weight[y == classes[0]] = n * len(probabilities) / n_negatives
        p_weight[y == classes[1]] = prevalence * len(probabilities) / n_positives

    cutpoints = np.array(range(1, n_quantiles + 1)) / n_quantiles * len(probabilities)

    quantiles = np.zeros(n_quantiles)
    accsum = 0
    j = 0
    for i in range(len(probabilities)):
        accsum = accsum + p_weight[i]
        if accsum < cutpoints[j]:
            quantiles[j] = quantiles[j] + probabilities[i] * p_weight[i]
        else:
            quantiles[j] = quantiles[j] + probabilities[i] * (p_weight[i] - (accsum - cutpoints[j]))
            withoutassign = accsum - cutpoints[j]
            while withoutassign > 0.1:
                j = j + 1
                assign = min(withoutassign, cutpoints[j] - cutpoints[j - 1])
                quantiles[j] = quantiles[j] + probabilities[i] * assign
                withoutassign = withoutassign - assign

    quantiles = quantiles / cutpoints[0]
    return quantiles


def is_V_shape(distance_func, mixture_func, test_distrib, step, verbose, **kwargs):
    """ Checks if the distance function is V-shaped

        Golden section search only works with V-shape distance (loss) functions

        Parameters
        ----------
        distance_func : function
            This is the loss function minimized during the search

        mixture_func : function
            The function used to generated the training mixture distribution given a value for the prevalence

        test_distrib : array
            The distribution of the positive class. The exact shape depends on the representation (pdfs, quantiles...)

        step: float
            The step to perform the linear search in the interval [0,1]

        verbose: bool
            True to print the distance for each prevalence and the best prevalence acording to the distance function

        kwargs : keyword arguments
            Here we pass the set of arguments needed by mixture functions: mixture_two_pdfs (for pdf-based classes) and
            compute quantiles (for quantiles-based classes). See the help of this two functions

        Returns
        -------
        True if the the distance function is v-shaped and False otherwise
    """
    n_mins = 0
    current_dist = distance_func(mixture_func(prevalence=0, **kwargs), test_distrib)
    next_dist = distance_func(mixture_func(prevalence=step, **kwargs), test_distrib)
    if verbose:
        print('%.2f %.4f # %.2f %.4f # ' % (0, current_dist, step, next_dist), end='')
    best_p = 2
    p = 2 * step
    while p <= 1:
        previous_dist = current_dist
        current_dist = next_dist
        next_dist = distance_func(mixture_func(prevalence=p, **kwargs), test_distrib)
        if verbose:
            print('%.2f %.4f # ' % (p, next_dist), end='')
        if current_dist <= previous_dist and current_dist <= next_dist:
            n_mins = n_mins + 1
            best_p = p
        p = p + step
    if verbose:
        print('\nNumber of minimuns: %d # Best prevalence: %.2f' % (n_mins, best_p))
    return n_mins <= 1


############
# Functions for solving ED-based methods
############
def solve_ed(G, a, C, b):
    """ Solves the optimization problem for ED-based quantifiers

        It just calls `quadprog.solve_qp` with the appropriate parameters. These paremeters were computed
        before by calling `compute_ed_param_train` and `compute_ed_param_test`.
        In the derivation of the optimization problem, the last class is put in terms of the rest of classes. Thus,
        we have to add 1-prevalences.sum() which it is the prevalence of the last class

        Parameters
        ----------
        G : array, shape (n_classes, n_classes)

        C : array, shape (n_classes, n_constraints)
            n_constraints will be n_classes + 1

        b : array, shape (n_constraints,)

        a : array, shape (n_classes, )

        G, C and b are computed by `compute_ed_param_train` and a by `compute_ed_param_test`

        Returns
        -------
        prevalences : array, shape=(n_classes, )
           Vector containing the predicted prevalence for each class

        References
        ----------
        Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
        Juan José del Coz: An analysis of quantification methods based on matching distributions

        Hideko Kawakubo, Marthinus Christoffel Du Plessis, and Masashi Sugiyama. 2016. Computationally efficient
        class-prior estimation under class balance change using energy distance. Transactions on Information
        and Systems 99, 1 (2016), 176–186.
    """
    sol = quadprog.solve_qp(G=G, a=a, C=C, b=b)
    prevalences = sol[0]
    # the last class was removed from the problem, its prevalence is 1 - the sum of prevalences for the other classes
    return np.append(prevalences, 1 - prevalences.sum())


def compute_ed_param_train(distance_func, train_distrib, classes, n_cls_i):
    """ Computes params related to the train distribution for solving ED-problems using `quadprog.solve_qp`

        Parameters
        ----------
        distance_func : function
            The function used to measure the distance between each pair of examples

        train_distrib : array, shape (n_bins * n_classes, n_classes)
            Represents the distribution of each class in the training set

        classes : ndarray, shape (n_classes, )
            Class labels

        n_cls_i: ndarray, shape (n_classes, )
            The number of examples of each class

        Returns
        -------
        K : array, shape (n_classes, n_classes)
            Average distance between each pair of classes in the training set

        G : array, shape (n_classes - 1, n_classes - 1)

        C : array, shape (n_classes - 1, n_constraints)
            n_constraints will be equal to the number of classes (n_classes)

        b : array, shape (n_constraints,)

        See references below for further details

        References
        ----------
        Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
        Juan José del Coz: An analysis of quantification methods based on matching distributions

        Hideko Kawakubo, Marthinus Christoffel Du Plessis, and Masashi Sugiyama. 2016. Computationally efficient
        class-prior estimation under class balance change using energy distance. Transactions on Information
        and Systems 99, 1 (2016), 176–186.
    """
    n_classes = len(classes)
    #  computing sum de distances for each pair of classes
    K = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        K[i, i] = distance_func(train_distrib[classes[i]], train_distrib[classes[i]]).sum()
        for j in range(i + 1, n_classes):
            K[i, j] = distance_func(train_distrib[classes[i]], train_distrib[classes[j]]).sum()
            K[j, i] = K[i, j]

    #  average distance
    K = K / np.dot(n_cls_i, n_cls_i.T)

    B = np.zeros((n_classes - 1, n_classes - 1))
    for i in range(n_classes - 1):
        B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
        for j in range(n_classes - 1):
            if j == i:
                continue
            B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

    #  computing the terms for the optimization problem
    G = 2 * B
    if not is_pd(G):
        G = nearest_pd(G)

    C = -np.vstack([np.ones((1, n_classes - 1)), -np.eye(n_classes - 1)]).T
    b = -np.array([1] + [0] * (n_classes - 1), dtype=np.float)

    return K, G, C, b


def compute_ed_param_test(distance_func, train_distrib, test_distrib, K, classes, n_cls_i):
    """ Computes params related to the test distribution for solving ED-problems using `quadprog.solve_qp`

        Parameters
        ----------
        distance_func : function
            The function used to measure the distance between each pair of examples

        train_distrib : array, shape (n_bins * n_classes, n_classes)
            Represents the distribution of each class in the training set

        test_distrib : array, shape (n_bins * n_classes, 1)
            Represents the distribution of the testing set

        K : array, shape (n_classes, n_classes)
            Average distance between each pair of classes in the training set

        classes : ndarray, shape (n_classes, )
            Class labels

        n_cls_i: ndarray, shape (n_classes, )
            The number of examples of each class

        Returns
        -------
        a : array, shape (n_classes, )
            Term a for solving optimization problems using `quadprog.solve_qp`

        See references below for further details

        References
        ----------
        Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
        Juan José del Coz: An analysis of quantification methods based on matching distributions

        Hideko Kawakubo, Marthinus Christoffel Du Plessis, and Masashi Sugiyama. 2016. Computationally efficient
        class-prior estimation under class balance change using energy distance. Transactions on Information
        and Systems 99, 1 (2016), 176–186.
    """
    n_classes = len(classes)
    Kt = np.zeros(n_classes)
    for i in range(n_classes):
        Kt[i] = distance_func(train_distrib[classes[i]], test_distrib).sum()

    Kt = Kt / (n_cls_i.squeeze() * float(len(test_distrib)))

    a = 2 * (- Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1])
    return a


# this functions are not used yet!
# def solve_mmy(train_dist, test_dist, n_classes):
#     c = np.hstack((np.ones(len(train_dist)),
#                    np.zeros(n_classes)))
#
#     A_ub = np.vstack((np.hstack((-np.eye(len(train_dist)), train_dist)),
#                       np.hstack((-np.eye(len(train_dist)), -train_dist))
#                       ))
#
#     b_ub = np.vstack((test_dist, -test_dist))
#
#     A_eq = np.hstack((np.zeros(len(train_dist)),
#                       np.ones(n_classes)))
#     A_eq = np.expand_dims(A_eq, axis=0)
#
#     b_eq = 1
#
#     x = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)['x']
#
#     p = x[-n_classes:]
#
#     return p
