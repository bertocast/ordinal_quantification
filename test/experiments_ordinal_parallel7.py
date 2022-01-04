import numpy as np
import glob
import os
import copy
import gc
import pandas as pd
import warnings
from datetime import datetime
from joblib import Parallel, delayed

from pandas.core.common import SettingWithCopyWarning
from imblearn.metrics import geometric_mean_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.exceptions import DataConversionWarning

from classify_and_count.cc import CC, PCC
from classify_and_count.ac import AC, PAC
from distribution_matching.energy import EDX, EDy
from distribution_matching.df import HDX, HDy

from estimators.frank_and_hall import FrankAndHallTreeClassifier, FrankAndHallMonotoneClassifier
from estimators.ordinal_ddag import DDAGClassifier
from ordinal.pdf import PDFOrdinaly
from ordinal.ac import ACOrdinal

from estimators.cross_validation import CV_estimator
from utils import create_bags_with_multiple_prevalence
from metrics.multiclass import mean_absolute_error
from metrics.ordinal import emd, emd_distances

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

pd.set_option('display.float_format', lambda x: '%.5f' % x)

warnings.simplefilter("ignore", DataConversionWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)

# configuration params
master_seed = 2032
n_bags = 300
n_reps = 10
n_folds = 20
n_jobs_fh = 1  # no paralelism for frank and hall

normalization = True
#option = "DECOMP(CV)"  # son los datos que tengo hasta ahora
option = "CV(DECOMP)"
# option = "CV(DECOMP(GS))"

estimator_grid = {
    "n_estimators": [10, 20, 40, 70, 100, 200, 250, 500],
    "max_depth": [1, 5, 10, 15, 20, 25, 30],
    "min_samples_leaf": [1, 2, 5, 10, 20]}


estimator = RandomForestClassifier(random_state=master_seed, n_estimators=5, class_weight='balanced')

# estimator_grid = {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
#                      #'solver': ["lbfgs", "liblinear"]
#                      }
#
# estimator = LogisticRegression(C=1, random_state=master_seed, max_iter=5000, verbose=False, multi_class='auto', solver='liblinear')

# methods = ["AC_L2","AC_Ord","CC","CvMy_Eu","CvMy_Ma","EDX", "EDy_Eu", "EDy_Ma", "FRI_HD", "FRI_L1", "FRI_L2",
#            "HDX","HDy", "PAC_L2","PCC","PDF_EMD", PDF_L2"] #"PDF_HD"

methods = ["AC_L2","AC_Ord","CC","EDX", "EDy_EMD","EDy_Eu","HDX","HDy", "PAC_L2","PCC","PDF_EMD", "PDF_L2"]

bins_gen=8
bins_pdf_l2=4
bins_pdf_emd=32
bins_hdx=8

columns = ['dataset', 'method', 'decomposer', 'repxbag', 'truth', 'predictions', 'mae', 'mse', 'emd']
# decomposers = ["Monotone", "FHTree", "DAG"]
decomposers = ["Monotone", "FHTree"]


t = datetime.now()  # hour, minute, year, day, month
tpo = str(t.day) + "-" + str(t.month) + "_" + str(t.hour) + ":" + str(t.minute)

os.mkdir("../results/" + tpo)
path = "../results/" + tpo + "/"
filename_out = path + "_results_v7_reglog" + option + "_" + str(n_reps) + "x" + str(n_bags) + "CV" + str(n_folds)


def main():
    dataset_files = [
         # "../datasets/ordinal/affairs_gago.csv",  # 9.949
         #  "../datasets/ordinal/auto.data.ord_chu.csv",  # 10.475
          # "../datasets/ordinal/ERA.csv",  # 12.290
         "../datasets/ordinal/ESL.csv",  # 5.395  -->5 clases: 3,4,5,6,7
         # "../datasets/ordinal/bostonhousing.ord_chu.csv",  # 33.853
         # "../datasets/ordinal/cement_strength_gago.csv",  # 44.142
         # "../datasets/ordinal/kinematics_gago.csv",  # 84.0114
         # "../datasets/ordinal/abalone.ord_chu.csv",  # 210.760
         # "../datasets/ordinal/californiahousing_gago.csv",  # 915.165
         #   "../datasets/ordinal/ailerons_gago.csv"   # 1.916.322
        # ]
        # server2
        #     "../datasets/ordinal/LEV.csv",  #11.023
        #     "../datasets/ordinal/stock.ord.csv",  # 53.455
        #     "../datasets/ordinal/SWD.csv",  #23.048
        # "../datasets/ordinal/wpbcancer.ord_chu.csv",  # 41.330
        #   "../datasets/ordinal/winequality-red_gago.csv",  # 87.078
        #   "../datasets/ordinal/winequality-white_gago_rev.csv",  #244.255
        #  "../datasets/ordinal/SkillCraft1_rev_7clases.csv",  # 424.998
        #  "../datasets/ordinal/SkillCraft1_rev_8clases.csv",  # 398.771
        #  "../datasets/ordinal/skill_gago.csv",  # 538.764
    ]
    # "../datasets/ordinal/fireman_example_gago.csv    #3990695
    # "../datasets/ordinal/sentTermMatrix.csv    #89423836

    # dataset_files = ["../datasets/iris.csv"]     # to compare results with multi-class quantification
    # dataset_files = ["../datasets/ordinal/lirios.csv"]
    #dataset_files = ["../datasets/ordinal/ERA.csv"]
    # dataset_files = ["../datasets/ordinal/lirios.csv",
    #                  "../datasets/ordinal/ERA.csv"]

    dataset_names = [os.path.split(name)[-1][:-4] for name in dataset_files]
    print("There are a total of {} datasets.".format(len(dataset_files)))

    Parallel(n_jobs=-1)(delayed(aux_parallel)(dataset_names, dataset_files, rep)
                        for rep in range(n_reps))

    emds = collect_result_ficheros_directorio()
    print(emds)
    return None


def aux_parallel(dataset_names, dataset_files, rep):
    # errors= []
    for dname, dfile in zip(dataset_names, dataset_files):
        current_seed = master_seed + rep
        # errors.append(train_on_a_dataset(dname, dfile, current_seed))
        train_on_a_dataset(dname, dfile, rep + 1, current_seed)
    return None


def normalize(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def load_data(dfile, seed):
    df = pd.read_csv(dfile, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(np.int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    if normalization == True:
        print("normalizying")
        X_train, X_test = normalize(X_train, X_test)
    return X_train, X_test, y_train, y_test


def load_data_ordinal(dfile, seed):
    panda = pd.read_csv(dfile, sep=';', header=0)
    columns = panda.values.shape[1]
    X = panda.values[:, 0:columns - 1]
    y = panda.values[:, columns - 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    if (normalization == True):
        print("normalizying")
        X_train, X_test = normalize(X_train, X_test)
    return X_train, X_test, y_train, y_test


def train_on_a_dataset(dname, dfile, rep, current_seed):
    # all_errors_df = pd.DataFrame(columns=columns)

    # generating training-test partition
    X_train, X_test, y_train, y_test = load_data_ordinal(dfile, current_seed)
    # X_train, X_test, y_train, y_test = load_data(dfile, current_seed) # solo para probar los lirios

    n_classes = len(np.unique(y_train))
    n_bags_dataset = n_bags #* n_classes

    print("*** Training over {}, rep {}, seed {}, bags {}".format(dname, rep, current_seed, n_bags_dataset))

    # classifiers are fitted by each object (all methods will use exactly the same predictions)
    # but they checked whether the estimator is already fitted (by a previous object) or not

    # estimator for estimating the testing distribution, GridSearchCV
    skf_test = StratifiedKFold(n_splits=3, shuffle=True, random_state=current_seed)
    gs = GridSearchCV(estimator, param_grid=estimator_grid, verbose=False, cv=skf_test,
                      scoring=make_scorer(geometric_mean_score), n_jobs=-1, iid=False)
    gs.fit(X_train, y_train)
    #clf_test = RandomForestClassifier(**gs.best_params_, random_state=master_seed, class_weight='balanced')
    clf_test = gs.best_estimator_

    #  estimator for estimating the training distribution
    folds = np.min([n_folds, np.min(np.unique(y_train, return_counts=True)[1])])
    skf_cv_train = StratifiedKFold(n_splits=folds, shuffle=True, random_state=master_seed)
    cv_train = CV_estimator(estimator=clf_test, cv=skf_cv_train)  # puede luego no usarse segun la opcion

    for decomposer in decomposers:
        # errors_df = train_on_a_decomposer(cv_train, clf_test, decomposer, X_train, y_train, X_test, y_test,
        #                                  current_seed, dname)
        # all_errors_df = all_errors_df.append(errors_df)

        train_on_a_decomposer(cv_train, clf_test, skf_cv_train, decomposer, X_train, y_train, X_test, y_test,
                              current_seed, dname, rep)

    return None


def train_on_a_decomposer(cv_train_, clf_test_, skf_cv_train, decomposer,
                          X_train, y_train, X_test, y_test, current_seed, dname, rep):
    print("* Training {} with {} rep {}".format(dname, decomposer, rep))

    the_errors_df = pd.DataFrame(columns=columns)

    est_train_ = cv_train_
    est_test_ = clf_test_

    #est_train_ = copy.deepcopy(cv_train_)
    #est_test_ = copy.deepcopy(clf_test_)

    if option == "DECOMP(CV)":
        print("Option", option)
        if decomposer == "Monotone":
            est_train = FrankAndHallMonotoneClassifier(est_train_, n_jobs=n_jobs_fh)
            est_test = FrankAndHallMonotoneClassifier(est_test_, n_jobs=n_jobs_fh)
        elif decomposer == "FHTree":
            est_train = FrankAndHallTreeClassifier(est_train_, n_jobs=n_jobs_fh)
            est_test = FrankAndHallTreeClassifier(est_test_, n_jobs=n_jobs_fh)
        elif decomposer == "DAG":
            est_train = DDAGClassifier(est_train_, n_jobs=n_jobs_fh)  # OPCION 1
            est_test = DDAGClassifier(est_test_, n_jobs=n_jobs_fh)
        elif decomposer == "DAG_LV":
            est_train = DDAGClassifier(est_train_, predict_method='winner_node', n_jobs=n_jobs_fh)
            est_test = DDAGClassifier(est_test_, predict_method='winner_node', n_jobs=n_jobs_fh)

    elif option == "CV(DECOMP)":
        print("Option", option)
        if decomposer == "Monotone":
            est_test = FrankAndHallMonotoneClassifier(est_test_, n_jobs=n_jobs_fh)
        elif decomposer == "FHTree":
            est_test = FrankAndHallTreeClassifier(est_test_, n_jobs=n_jobs_fh)
        elif decomposer == "DAG":
            est_test = DDAGClassifier(est_test_, n_jobs=n_jobs_fh)
        elif decomposer == "DAG_LV":
            est_test = DDAGClassifier(est_test_, predict_method='winner_node', n_jobs=n_jobs_fh)

        est_train = CV_estimator(estimator=est_test, cv=skf_cv_train)

    elif option == "CV(DECOMP(GS))":
        print("Option", option)
        # skf_train = StratifiedKFold(n_splits=3, shuffle=True, random_state=current_seed)
        # gs_train = GridSearchCV(estimator, param_grid=estimator_grid, verbose=False, cv=skf_train,
        #                         scoring=make_scorer(geometric_mean_score), n_jobs=-1, iid=False)
        #
        # if decomposer == "Monotone":
        #     est_test = FHMonotone(est_test_, n_jobs=n_jobs_fh)
        #     est_train_gs = FHMonotone(gs_train, n_jobs=n_jobs_fh)
        # elif decomposer == "FHTree":
        #     est_test = FHTree(est_test_, n_jobs=n_jobs_fh)
        #     est_train_gs = FHTree(gs_train, n_jobs=n_jobs_fh)
        # elif decomposer == "DAG":
        #     est_test = DAGClassifier(est_test_, n_jobs=n_jobs_fh)
        #     est_train_gs = DAGClassifier(gs_train, n_jobs=n_jobs_fh)
        # elif decomposer == "DAG_LV":
        #     est_test = DAGClassifier(est_test_, predict_method='leaves_probabilistic', n_jobs=n_jobs_fh)
        #     est_train_gs = DAGClassifier(gs_train, predict_method='leaves_probabilistic', n_jobs=n_jobs_fh)
        #
        # est_train = CV_estimator(estimator=est_train_gs, cv=skf_train, n_jobs=-1)


    if "AC" in methods:
        ac = AC(estimator_train=est_train, estimator_test=est_test)
        ac.fit(X_train, y_train)
    if "AC_HD" in methods:
        ac_hd = AC(estimator_train=est_train, estimator_test=est_test, distance='HD')
        ac_hd.fit(X_train, y_train)
    if "AC_L1" in methods:
        ac_l1 = AC(estimator_train=est_train, estimator_test=est_test, distance='L1')
        ac_l1.fit(X_train, y_train)
    if "AC_L2" in methods:
        ac_l2 = AC(estimator_train=est_train, estimator_test=est_test, distance='L2')
        ac_l2.fit(X_train, y_train)
    if "AC_Ord" in methods:
        ac_ord = ACOrdinal(estimator_train=est_train, estimator_test=est_test)
        ac_ord.fit(X_train, y_train)
    if "CC" in methods:
        cc = CC(estimator_test=est_test)
        cc.fit(X_train, y_train)
    if "EDX" in methods:
        edx = EDX()
        edx.fit(X_train, y_train)
    if "EDy_EMD" in methods:
        edy_emd = EDy(estimator_train=est_train, estimator_test=est_test, distance=emd_distances)
        edy_emd.fit(X_train, y_train)
    if "EDy_Eu" in methods:
        edy_eu = EDy(estimator_train=est_train, estimator_test=est_test, distance=euclidean_distances)
        edy_eu.fit(X_train, y_train)
    if "EDy_Ma" in methods:
        edy_ma = EDy(estimator_train=est_train, estimator_test=est_test)
        edy_ma.fit(X_train, y_train)
    if "HDX" in methods:
        hdx = HDX(n_bins=bins_hdx)
        hdx.fit(X_train, y_train)
    if "HDy" in methods:
        hdy = HDy(estimator_train=est_train, estimator_test=est_test, n_bins=bins_gen) #default n_bins=8
        hdy.fit(X_train, y_train)
    if "PAC" in methods:
        pac = PAC(estimator_train=est_train, estimator_test=est_test)
        pac.fit(X_train, y_train)
    if "PAC_HD" in methods:
        pac_hd = PAC(estimator_train=est_train, estimator_test=est_test, distance='HD')
        pac_hd.fit(X_train, y_train)
    if "PAC_L1" in methods:
        pac_l1 = PAC(estimator_train=est_train, estimator_test=est_test, distance='L1')
        pac_l1.fit(X_train, y_train)
    if "PAC_L2" in methods:
        pac_l2 = PAC(estimator_train=est_train, estimator_test=est_test, distance='L2')
        pac_l2.fit(X_train, y_train)
    if "PCC" in methods:
        pcc = PCC(estimator_test=est_test)
        pcc.fit(X_train, y_train)
    if "PDF_EMD" in methods:
        pdf_emd = PDFOrdinaly(estimator_train=est_train, estimator_test=est_test, distance='EMD', n_bins=bins_pdf_emd)
        pdf_emd.fit(X_train, y_train)
    if "PDF_HD" in methods:
        pdf_hd = PDFOrdinaly(estimator_train=est_train, estimator_test=est_test, distance='HD', n_bins=bins_gen)
        pdf_hd.fit(X_train, y_train)
    if "PDF_L2" in methods:
        pdf_l2 = PDFOrdinaly(estimator_train=est_train, estimator_test=est_test, distance='L2', n_bins=bins_pdf_l2)
        pdf_l2.fit(X_train, y_train)

    n_classes = len(np.unique(y_train))
    #n_bags_dataset = n_bags * n_classes
    n_bags_dataset = n_bags

    for n_bag, (X_test_, y_test_, prev_true) in enumerate(
            create_bags_with_multiple_prevalence(X_test, y_test, n_bags_dataset, current_seed)):

        # for X_test_ in [X_test]:
        #     unique_elements, counts_elements = np.unique(y_test, return_counts=True)
        #     prev_true = counts_elements / float(len(y_test))  #sum(counts_elements)

        #print("bag", n_bag)

        prev_preds = []

        if "AC" in methods: prev_preds.append(ac.predict(X_test_))
        if "AC_HD" in methods: prev_preds.append(ac_hd.predict(X_test_))
        if "AC_L1" in methods: prev_preds.append(ac_l1.predict(X_test_))
        if "AC_L2" in methods: prev_preds.append(ac_l2.predict(X_test_))
        if "AC_Ord" in methods: prev_preds.append(ac_ord.predict(X_test_))
        if "CC" in methods: prev_preds.append(cc.predict(X_test_))
        if "EDX" in methods: prev_preds.append(edx.predict(X_test_))
        if "EDy_EMD" in methods: prev_preds.append(edy_emd.predict(X_test_))
        if "EDy_Eu" in methods: prev_preds.append(edy_eu.predict(X_test_))
        if "EDy_Ma" in methods: prev_preds.append(edy_ma.predict(X_test_))
        if "HDX" in methods: prev_preds.append(hdx.predict(X_test_))
        if "HDy" in methods: prev_preds.append(hdy.predict(X_test_))
        if "PAC" in methods: prev_preds.append(pac.predict(X_test_))
        if "PAC_HD" in methods: prev_preds.append(pac_hd.predict(X_test_))
        if "PAC_L1" in methods: prev_preds.append(pac_l1.predict(X_test_))
        if "PAC_L2" in methods: prev_preds.append(pac_l2.predict(X_test_))
        if "PCC" in methods: prev_preds.append(pcc.predict(X_test_))
        if "PDF_EMD" in methods: prev_preds.append(pdf_emd.predict(X_test_))
        if "PDF_HD" in methods: prev_preds.append(pdf_hd.predict(X_test_))
        if "PDF_L2" in methods: prev_preds.append(pdf_l2.predict(X_test_))

        for n_method, (method, prev_pred) in enumerate(zip(methods, prev_preds)):
            # for n_method, (method, prev_pred) in enumerate(zip(["AC", "CC", "EDX", "EDy", "HDX", "HDy"], prev_preds)):
            mae_ = mean_absolute_error(prev_true, prev_pred)
            mse_ = mean_squared_error(prev_true, prev_pred)
            emd_ = emd(prev_true, prev_pred)
            rb = str(rep) + "x" + str(n_bag)
            # print("prev_true, prev_pred, mae, mse, emd", prev_true, prev_pred, mae_, mse_, emd_)
            # pd_aux = pd.DataFrame([[dname, method, decomposer, rb, prev_true, prev_pred, mae_, mse_, emd_]],
            #                  columns=columns)
            pd_aux = pd.DataFrame([[dname, method, decomposer, rb, prev_true, prev_pred, mae_, mse_, emd_]],
                             columns=columns)
            the_errors_df = the_errors_df.append(pd_aux)

    # the_errors_df.to_csv(filename_out + "_"+ decomposer + ".csv", mode='a', index=None)
    the_errors_df.to_csv(filename_out + "_" + decomposer + "_" + dname + ".csv", mode='a', index=None)
    return None


def collect_result_ficheros_directorio():
    # concatenate all files
    all_files = glob.glob(path + "_result*.csv")
    all_files.sort()
    n_files = len(all_files) // len(decomposers)
    print("**** Procesing", n_files, "datasets")
    good_cols = ['dataset', 'method', 'decomposer', 'mae', 'mse', 'emd']  # 'repxbag', 'truth', 'predictions',
    l = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, usecols=good_cols, header=0)
        df = df[df['dataset'] != "dataset"]  # removing lines with duplicated columns
        l.append(df)
        print("* Reading", filename, len(df), "bags")

    res_df = pd.concat(l, axis=0, ignore_index=True)
    # converting errors to float
    res_df = res_df.astype(dtype={'mae': 'float', 'mse': 'float', 'emd': 'float'})
    # res_df.to_csv("_tmp.csv", mode='w', index=None)  #Chapuza
    # res_df = pd.read_csv("_tmp.csv")

    fout = path + "__means_" + option + "_" + str(n_reps) + "x" + str(n_bags) + "CV" + str(n_folds) \
           + "_" + str(n_files) + "_dts" + ".csv"

    cont = 0
    for decomposer in decomposers:
        results_df = res_df[res_df['decomposer'] == decomposer].sort_values(by=['dataset'])
        if len(results_df) == 0:
            continue
        columns = ['emd', 'mae', 'mse']  # 'dataset' es el index
        for error in columns:
            means_df = results_df.groupby(['decomposer', 'dataset', 'method'])[[error]].agg(["mean"]).unstack().round(5)
            means_df.columns = methods
            means_df['error'] = error  # añado una columna al final
            if cont == 0:
                means_df.to_csv(fout, mode='w')
            else:
                means_df.to_csv(fout, mode='a', header=False)
            cont += 1

    if cont > 0:
        emds_df = pd.read_csv(fout)
    else:
        emds_df = None
    return emds_df


if __name__ == '__main__':
    main()
