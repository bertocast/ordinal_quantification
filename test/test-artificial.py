import numpy as np
import matplotlib.pyplot as plt
import numbers

from sklearn.utils import check_X_y
from sklearn.metrics.pairwise import euclidean_distances

from base import UsingClassifiers

from classify_and_count.cc import CC, PCC
from classify_and_count.ac import AC, PAC
from ordinal.ac import ACOrdinal
from distribution_matching.energy import EDX, EDy, CvMy
from distribution_matching.df import HDX, HDy
from ordinal.pdf import PDFOrdinaly
from estimators import CV_estimator, FrankAndHallMonotoneClassifier, FrankAndHallTreeClassifier
from metrics.ordinal import emd, emd_distances

from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression

import time


def create_bags(x, y, n=1001, randomg=None):
    if isinstance(randomg, (numbers.Integral, np.integer)):
        randomg = np.random.RandomState(randomg)
    if not isinstance(randomg, np.random.RandomState):
        raise ValueError("Invalid random generaror object")

    x, y = check_X_y(x, y)
    classes = np.unique(y)
    nclasses = len(classes)
    nexamples = len(x)

    for i in range(n):
        # Kraemer method:

        # to soft limits
        low = round(nexamples * 0.05)
        high = round(nexamples * 0.95)

        ps = randomg.randint(low, high, nclasses - 1)
        ps = np.append(ps, [0, nexamples])
        ps = np.diff(np.sort(ps))  # number of samples for each class
        prev = ps / nexamples  # to obtain prevalences
        indexes = []
        for n, p in zip(classes, ps.tolist()):
            if p != 0:
                idx = randomg.choice(np.where(y == n)[0], p, replace=True)
                indexes.append(idx)

        indexes = np.concatenate(indexes)
        yield x[indexes], y[indexes], prev, indexes


# MAIN
n_classes = 5
nbags = 300  # 50 * n_classes
mu_sep = 3  # 3
sigma = 1.5

seed = 42
rng = np.random.RandomState(seed)

estimator_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
estimator = LogisticRegression(random_state=seed, max_iter=1000, solver="liblinear")

decomposer = "Monotone"
#  decomposer = "FHTree"

values = [50, 100, 200, 500, 1000, 2000]

nreps = 10

#   create objects
cc = CC()
pcc = PCC()
edx = EDX()
hdx = HDX(n_bins=16)
ac = AC(distance='L2')
acord = ACOrdinal()
cvmy_euc = CvMy(distance=euclidean_distances)
edy_emd = EDy(distance=emd_distances)
edy_euc = EDy(distance=euclidean_distances)
hdy = HDy(n_bins=8)
pac = PAC()
pdfordy_emd = PDFOrdinaly(n_bins=32, distance='EMD')
pdfordy_l2 = PDFOrdinaly(n_bins=4, distance='L2')

#   methods
methods = [ac, acord, cc, cvmy_euc, edx, edy_emd, edy_euc, hdx, hdy, pac, pcc, pdfordy_emd, pdfordy_l2]
methods_names = ['AC', 'ACOrd', 'CC', 'CvMY_euc',
                 'EDX', 'EDy_emd', 'EDy_euc', 'HDX', 'HDy', 'PAC', 'PCC', 'PDFOrdy_emd', 'PDFOrdy_l2']
markers = ['-s', '--s', '-x', '--^', '-*', '-o', '--o', '--*', '-D', ':s', '--x', '--D', ':D']

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

            mu = mu_sep * (i + 1)  # mu of class i

            examples_class_i = sigma * rng.randn(n_train) + mu  # training examples of class i

            X_train = np.append(X_train, examples_class_i)
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

        # fitting classifiers here, all methods use exactly the same predictions!
        skf_test = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        gs = GridSearchCV(estimator, param_grid=estimator_grid, verbose=False, cv=skf_test,
                          scoring=make_scorer(geometric_mean_score), n_jobs=None, iid=False)
        gs.fit(X_train, y_train_binary)
        best_lr = gs.best_estimator_

        # estimator for estimating the training distribution, CV 20
        folds = 20
        skf_train = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        estimator_train = None
        if decomposer == "Monotone":
            estimator_train = CV_estimator(estimator=FrankAndHallMonotoneClassifier(estimator=best_lr, n_jobs=None),
                                           n_jobs=None, cv=skf_train)
        elif decomposer == "FHTree":
            estimator_train = CV_estimator(estimator=FrankAndHallTreeClassifier(estimator=best_lr, n_jobs=None),
                                           n_jobs=None, cv=skf_train)

        # estimator_train = FrankAndHallMonotoneClassifier(estimator=estimator, n_jobs=None)
        estimator_train.fit(X_train, y_train)

        predictions_train = estimator_train.predict_proba(X_train)

        for nmethod, method in enumerate(methods):
            if isinstance(method, UsingClassifiers):
                method.fit(X=X_train, y=y_train, predictions_train=predictions_train)
            else:
                method.fit(X=X_train, y=y_train)

        estimator_test = None
        if decomposer == "Monotone":
            estimator_test = FrankAndHallMonotoneClassifier(estimator=best_lr, n_jobs=-1)
        elif decomposer == "FHTree":
            estimator_test = FrankAndHallTreeClassifier(estimator=best_lr, n_jobs=-1)

        estimator_test.fit(X_train, y_train)

        predictions_test = estimator_test.predict_proba(X_test)

        print('Test', end=' ')
        for n_bag, (pred_test_, y_test_, prev_true, idxs) in enumerate(
                create_bags(x=predictions_test, y=y_test, n=nbags, randomg=seed)):

            for nmethod, method in enumerate(methods):

                # print(nmethod+1, end='')

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

    print('Time: ', execution_times/(nreps * nbags))

    name_file = "../results/allresults-test-artificial-sep" + str(mu_sep) + "-rep" + str(nreps) + "-value" + str(
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

    file_all.close()


results = results / (nreps * nbags)


name_file = "../results/avgresults-test-artificial-sep" + str(mu_sep) + "-rep" + str(nreps) + ".txt"

file_avg = open(name_file, 'w')
for index, m in enumerate(methods_names):
    file_avg.write('MEAN %6s:' % m)
    for i in results[index, :]:
        file_avg.write('%.5f\t' % i)
    file_avg.write('\n')

file_avg.close()

print('', flush=True)


fig1 = plt.figure(1)
for index, m in enumerate(methods_names):
    plt.plot(values, results[index, :], markers[index], color='k', markersize=4, linewidth=1, label=m)

plt.legend()
plt.show()
name_file = "../results/fig_test_mu_sep_" + str(mu_sep) + ".png"
plt.savefig(name_file)
plt.close()
