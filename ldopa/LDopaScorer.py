from __future__ import print_function, division
import numpy as np
import pandas as pd
import synapseclient as sc
import argparse
import os
from nonLinearInterpAUPRC import getAUROC_PR
from sklearn.metrics.base import _average_binary_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize

syn = sc.login()

TRAINING_TABLE = "syn10495809"
TESTING_TABLE = "syn10518012"
INDEX_COL = "dataFileHandleId"
SUBMISSIONS_SAVED = {
                    'tremor': 'syn11023745',
                    'dyskinesia': 'syn11023753',
                    'bradykinesia': 'syn11023754'}
CATEGORY_WEIGHTS = {
        'tremor': [896, 381, 214, 9, 0],
        'dyskinesia': [531, 129],
        'bradykinesia': [990, 419]}
SUBMISSION_TEMPLATES = {
    'tremor': 'syn11315931',
    'dyskinesia': 'syn11315929',
    'bradykinesia': 'syn11315927'}
PREDICTIONS_SAVED = {
                    'tremor': 'syn11472596',
                    'dyskinesia': 'syn11472597',
                    'bradykinesia': 'syn11472598'}


def read_args():
    parser = argparse.ArgumentParser(
            description="Score an L-Dopa Submission file (AUPRC).")
    parser.add_argument("phenotype",
                        help="One of 'tremor', 'dyskinesia', or 'bradykinesia'")
    parser.add_argument("submission",
                        help="filepath to submissions file")
    args = parser.parse_args()
    return args


def read_data(path, phenotype):
    df = pd.read_csv(path, header=0) if isinstance(path, str) else path
    df = df.set_index(df.columns[0], drop=True)
    if "bscore" in df.columns: df.drop("bscore", axis=1, inplace=True)

    train_table = get_table(TRAINING_TABLE)
    test_table = get_table(TESTING_TABLE)

    train = df.join(
            train_table[['tremorScore', 'dyskinesiaScore', 'bradykinesiaScore']],
            how='inner')
    test = df.join(
            test_table[['tremorScore', 'dyskinesiaScore', 'bradykinesiaScore']],
            how='inner')

    submission_template = syn.get(SUBMISSION_TEMPLATES[phenotype])
    submission_template = pd.read_csv(
            submission_template.path, index_col='dataFileHandleId')
    to_keep = submission_template.index

    train = train.loc[to_keep.intersection(train.index)]
    test = test.loc[to_keep.intersection(test.index)]

    train_X = train.drop(
        ['tremorScore', 'dyskinesiaScore', 'bradykinesiaScore'], axis=1).values
    test_X = test.drop(
        ['tremorScore', 'dyskinesiaScore', 'bradykinesiaScore'], axis=1)

    train_y = train["{}Score".format(phenotype)].values
    test_y = test["{}Score".format(phenotype)]

    return train_X, test_X, train_y, test_y, to_keep


def get_table(synId):
    q = syn.tableQuery("select * from {}".format(synId))
    df = q.asDataFrame()
    df = df.set_index(INDEX_COL, drop=True)
    return df


def train_ensemble(X, y):
    rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=500))
    lr = OneVsRestClassifier(LogisticRegressionCV())
    svm = OneVsRestClassifier(SVC(probability=True))
    ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr), ('svm', svm)], voting='soft')
    ensemble.fit(X, y)
    return ensemble


def pr_auc_score(y_true, y_score, average='micro', sample_weight=None):
    def _binary_pr_auc_score(y_true, y_score, sample_weight=None):
        if len(np.unique(y_true)) != 2:
            raise ValueError("Only one class present in y_true. AUPRC score "
                             "is not defined in that case.")
        fpr, tpr, thresholds = precision_recall_curve(y_true, y_score,
                                        sample_weight=sample_weight)
        return auc(tpr, fpr, reorder=False)
    return _average_binary_score(
        _binary_pr_auc_score, y_true, y_score, average,
        sample_weight=sample_weight)


def get_pr_auc_scores(X, y, y_labels, clf, average='micro'):
    assert all([i == j for i, j in zip(y_labels, clf.classes_)])
    n_classes = len(y_labels)
    if n_classes > 2:
        y_true = label_binarize(y, y_labels)
    else:
        y_true = y
    y_score = clf.predict_proba(X) if n_classes > 2 else clf.predict_proba(X).T[1]
    aupr = pr_auc_score(y_true, y_score, average=average)
    return aupr, y_score, y_true


def getNonLinearInterpAupr(X, y, y_labels, clf):
    n_classes = len(y_labels)
    if n_classes > 2:
        y_true = label_binarize(y, y_labels)
    else:
        y_true = y
    y_score = clf.predict_proba(X) if n_classes > 2 else clf.predict_proba(X).T[1]
    return nonLinearInterpAupr(y_score, y_true)


def nonLinearInterpAupr(y_score, y_true):
    if y_score.ndim > 1:
        sub_stats = pd.DataFrame(np.concatenate((y_score, y_true), axis=1),
                dtype='float64')
    else:
        sub_stats = pd.DataFrame(np.array([y_score, y_true]).T)
    results = getAUROC_PR(sub_stats)
    return results, y_score, y_true


def writeSubmissionToFile(phenotype, submission, submissionId):
    write_location = SUBMISSIONS_SAVED[phenotype]
    f = sc.File(path=submission, name="{}.csv".format(submissionId),
                parent=write_location)
    syn.store(f)


def writePredictionsToFile(phenotype, index, predictions, submissionId):
    write_location = PREDICTIONS_SAVED[phenotype]
    pd.DataFrame(predictions, index=index).to_csv(
            "/home/ubuntu/LDopaScoring/predictions/{}.csv".format(submissionId),
            index=True, header=False)
    f = sc.File(path="/home/ubuntu/LDopaScoring/predictions/{}.csv".format(submissionId),
        name="{}.csv".format(submissionId), parent=write_location,
                             annotations={'submissionId': submissionId})
    syn.store(f)


def getWeightedMean(phenotype, scores):
    numer = 0
    denom = sum(CATEGORY_WEIGHTS[phenotype])
    for w, s in zip(CATEGORY_WEIGHTS[phenotype], scores):
        if pd.notnull(s):
            numer += w * s
    return numer / denom


def calculatePVal(y_score, y_true, trueAupr, phenotype):
    auprs = []
    n_iterations = 10
    for i in range(n_iterations):
        np.random.shuffle(y_score)
        results = nonLinearInterpAupr(y_score, y_true)
        if phenotype == 'tremor':
            weighted_results = getWeightedMean(phenotype, results)
        else:
            weighted_results = results[0]
        auprs.append(weighted_results)
    return sum([a > trueAupr for a in auprs]) / n_iterations


def score(phenotype, submission, submissionId=None):
    #if submissionId: writeSubmissionToFile(phenotype, submission, submissionId)
    train_X, test_X, train_y, test_y, index = read_data(submission, phenotype)
    ensemble = train_ensemble(train_X, train_y)
    results, y_score, y_true = getNonLinearInterpAupr(test_X, test_y,
            np.arange(len(CATEGORY_WEIGHTS[phenotype])), ensemble)
    #X = np.append(train_X, test_X, axis=0)
    #y = np.append(train_y, test_y, axis=0)
    #results, y_score, y_true = getNonLinearInterpAupr(X, y,
    #        np.arange(len(CATEGORY_WEIGHTS[phenotype])), ensemble)
    #if submissionId: writePredictionsToFile(phenotype, index, y_score, submissionId)
    if phenotype == 'tremor':
        weighted_aupr = getWeightedMean(phenotype, results)
    else:
        weighted_aupr = results[0]
    #pval = calculatePVal(y_score, y_true, weighted_aupr, phenotype)
    return weighted_aupr


def main():
    args = read_args()
    aupr = score(args.phenotype, args.submission)
    print(aupr)
    return aupr


if __name__ == "__main__":
    main()
