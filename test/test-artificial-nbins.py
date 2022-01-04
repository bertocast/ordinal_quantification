import numpy as np
import numbers

from sklearn.utils import check_X_y

from base import UsingClassifiers

from distribution_matching.df import HDX, HDy
from ordinal.pdf import PDFOrdinaly
from estimators import CV_estimator, FrankAndHallMonotoneClassifier, FrankAndHallTreeClassifier

from metrics.ordinal import emd

from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression

import time


def create_bags(X, y, n=1001, rng=None):
    if isinstance(rng, (numbers.Integral, np.integer)):
        rng = np.random.RandomState(rng)
    if not isinstance(rng, np.random.RandomState):
        raise ValueError("Invalid random generaror object")

    X, y = check_X_y(X, y)
    classes = np.unique(y)
    n_classes = len(classes)
    m = len(X)

    for i in range(n):
        # Kraemer method:

        # to soft limits
        low = round(m * 0.05)
        high = round(m * 0.95)

        ps = rng.randint(low, high, n_classes - 1)
        ps = np.append(ps, [0, m])
        ps = np.diff(np.sort(ps))  # number of samples for each class
        prev = ps / m  # to obtain prevalences
        idxs = []
        for n, p in zip(classes, ps.tolist()):
            if p != 0:
                idx = rng.choice(np.where(y == n)[0], p, replace=True)
                idxs.append(idx)

        idxs = np.concatenate(idxs)
        yield X[idxs], y[idxs], prev, idxs



# MAIN
n_classes = 5
nbags = 300 # 50 * n_classes
mu_sep = 3
sigma = 1.5

seed = 42
rng = np.random.RandomState(seed)

estimator_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
estimator = LogisticRegression(random_state=seed, max_iter=1000, solver="liblinear")

decomposer = "Monotone"
# decomposer = "FHTree"

values = [50, 100, 200, 500, 1000, 2000]

nreps = 10

#   create objects
hdx_4bins = HDX(n_bins=4)
hdy_4bins = HDy(n_bins=4)
pdf_emd_4bins = PDFOrdinaly(n_bins=4, distance='EMD')
pdf_l2_4bins = PDFOrdinaly(n_bins=4, distance='L2')

hdx_8bins = HDX(n_bins=8)
hdy_8bins = HDy(n_bins=8)
pdf_emd_8bins = PDFOrdinaly(n_bins=8, distance='EMD')
pdf_l2_8bins = PDFOrdinaly(n_bins=8, distance='L2')

hdx_16bins = HDX(n_bins=16)
hdy_16bins = HDy(n_bins=16)
pdf_emd_16bins = PDFOrdinaly(n_bins=16, distance='EMD')
pdf_l2_16bins = PDFOrdinaly(n_bins=16, distance='L2')

hdx_32bins = HDX(n_bins=32)
hdy_32bins = HDy(n_bins=32)
pdf_emd_32bins = PDFOrdinaly(n_bins=32, distance='EMD')
pdf_l2_32bins = PDFOrdinaly(n_bins=32, distance='L2')

hdx_64bins = HDX(n_bins=64)
hdy_64bins = HDy(n_bins=64)
pdf_emd_64bins = PDFOrdinaly(n_bins=64, distance='EMD')
pdf_l2_64bins = PDFOrdinaly(n_bins=64, distance='L2')

#   methods
methods = [hdx_4bins, hdx_8bins, hdx_16bins, hdx_32bins, hdx_64bins,
           hdy_4bins, hdy_8bins, hdy_16bins, hdy_32bins, hdy_64bins,
           pdf_emd_4bins, pdf_emd_8bins, pdf_emd_16bins, pdf_emd_32bins, pdf_emd_64bins,
           pdf_l2_4bins, pdf_l2_8bins, pdf_l2_16bins, pdf_l2_32bins, pdf_l2_64bins]
methods_names = ['HDX_4b', 'HDX_8b', 'HDX_16b', 'HDX_32b', 'HDX_64b',
                 'HDy_4b', 'HDy_8b', 'HDy_16b', 'HDy_32b', 'HDy_64b',
                 'PDF_emd_4b', 'PDF_emd_8b', 'PDF_emd_16b', 'PDF_emd_32b', 'PDF_emd_64b',
                 'PDF_l2_4b', 'PDF_l2_8b', 'PDF_l2_16b', 'PDF_l2_32b', 'PDF_l2_64b']
#   to store all the results
results = np.zeros((len(methods), len(values)))


for k in range(len(values)):

    print()
    print('Train examples: ', values[k], end='')
    n_train = values[k]
    n_test = 2000  # n_train

    all_results = np.zeros((len(methods), nreps * nbags))
    execution_times = np.zeros(len(methods))

    for rep in range(nreps):

        print()
        print('Rep#', rep+1, end=' ')
        # create datasets
        X_train = []
        y_train = []
        y_train_binary = []
        X_test = []
        y_test = []
        for i in range(n_classes):
            mu = mu_sep * (i + 1)

            training_examples = sigma * rng.randn(n_train) + mu

            X_train = np.append(X_train, training_examples)
            y_train = np.append(y_train, (i + 1) * np.ones(n_train))
            if i < n_classes / 2:
                y_train_binary = np.append(y_train_binary, -np.ones(n_train))
            else:
                y_train_binary = np.append(y_train_binary, np.ones(n_train))

            testing_examples = sigma * rng.randn(n_test) + mu

            X_test = np.append(X_test, testing_examples)
            y_test = np.append(y_test, (i + 1) * np.ones(n_test))

        X_train = X_train.reshape(-1, 1)
        # y_train = y_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        # y_test = y_test.reshape(-1, 1)

        #  fitting classifiers here, all methods use exactly the same predictions!
        skf_test = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        gs = GridSearchCV(estimator, param_grid=estimator_grid, verbose=False, cv=skf_test,
                          scoring=make_scorer(geometric_mean_score), n_jobs=None, iid=False)
        gs.fit(X_train, y_train_binary)
        best_lr = gs.best_estimator_

        #  estimator for estimating the training distribution, CV 50
        folds = 20 # np.min([50, np.min(np.unique(y_train, return_counts=True)[1])])
        skf_train = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        if decomposer == "Monotone":
            estimator_train = CV_estimator(estimator=FrankAndHallMonotoneClassifier(estimator=best_lr, n_jobs=None),
                                           n_jobs=None, cv=skf_train)
        elif decomposer == "FHTree":
            estimator_train = CV_estimator(estimator=FrankAndHallTreeClassifier(estimator=best_lr, n_jobs=None),
                                           n_jobs=None, cv=skf_train)

        estimator_train.fit(X_train, y_train)

        predictions_train = estimator_train.predict_proba(X_train)

        for nmethod, method in enumerate(methods):
            if isinstance(method, UsingClassifiers):
                method.fit(X=X_train, y=y_train, predictions_train=predictions_train)
            else:
                method.fit(X=X_train, y=y_train)

        if decomposer == "Monotone":
            estimator_test = FrankAndHallMonotoneClassifier(estimator=best_lr, n_jobs=-1)
        elif decomposer == "FHTree":
            estimator_test = FrankAndHallTreeClassifier(estimator=best_lr, n_jobs=-1)

        estimator_test.fit(X_train, y_train)

        predictions_test = estimator_test.predict_proba(X_test)

        print('Test', end=' ')
        for n_bag, (pred_test_, y_test_, prev_true, idxs) in enumerate(
                create_bags(X=predictions_test, y=y_test, n=nbags, rng=seed)):

            for nmethod, method in enumerate(methods):

                # print(nmethod+1, end='')

                t = time.process_time()
                if isinstance(method, UsingClassifiers):
                    p_predicted = method.predict(X=None, predictions_test=pred_test_)
                else:
                    p_predicted = method.predict(X=X_test[idxs])

                elapsed_time = time.process_time()
                execution_times[nmethod] = execution_times[nmethod] + elapsed_time - t

                error = emd(prev_true, p_predicted)
                all_results[nmethod, rep * nbags + n_bag] = error
                results[nmethod, k] = results[nmethod, k] + error

    print('Time: ', execution_times/(nreps * nbags) )

    name_file = "../results/allresults-test-bin-artificial-sep" + str(mu_sep) + "-rep" + str(nreps) + "-value" + str(
        values[k]) + "_" + decomposer + ".txt"
    file_all = open(name_file, 'w')

    for method_name in methods_names:
        file_all.write('%s,' % method_name)
    file_all.write('\n')
    for nrep in range(nreps):
        for n_bag in range(nbags):
            for n_method in range(len(methods_names)):
                file_all.write('%.5f, ' % all_results[n_method, nrep * nbags + n_bag])
            file_all.write('\n')

    all_results = np.zeros((len(methods) * nreps * nbags, 2))

    file_all.close()


results = results / (nreps * nbags)


name_file = "../results/avgresults-test-bin-artificial-sep" + str(mu_sep) + "-rep" + str(nreps) + "_" + decomposer + ".txt"
file_avg = open(name_file, 'w')
for index, m in enumerate(methods_names):
    file_avg.write('MEAN %6s:' % m)
    for i in results[index, :]:
        file_avg.write('%.5f\t' % i)
    file_avg.write('\n')

file_avg.close()






