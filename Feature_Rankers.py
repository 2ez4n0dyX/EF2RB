# Functions to calculate entropy, joint entropy, conditional entropy, mutual information, and redundancy.
# mrmr function for minimum redundancy maximum relevance feature selection.
# AFFMI for calculating average feature-feature mutual information.
# JMI and AJMII for joint mutual information.
# jomic for joint mutual information criterion-based selection.
# Functions for intrinsic information and gain ratio calculation.
# Gratio for gain ratio-based selection.
# PCFS for Pearson correlation-based feature selection.
# MIFSND for MIFSND feature selection method.



import numpy as np
import pandas as pd

def entropy(Y):
    _, counts = np.unique(Y, return_counts=True, axis=0)
    probabilities = counts / len(Y)
    return -np.sum(probabilities * np.log2(probabilities))

def joint_entropy(Y, X):
    YX = np.c_[Y, X]
    return entropy(YX)

def conditional_entropy(Y, X):
    return joint_entropy(Y, X) - entropy(X)

def mutual_information(Y, X):
    return entropy(Y) - conditional_entropy(Y, X)

def redundancy(Xi, Xj):
    mutual_infos = [mutual_information(Xi, Xj.iloc[:, j]) for j in range(Xj.shape[1])]
    return np.mean(mutual_infos)

def mrmr(X, y, no_features='auto'):
    selected_features = pd.DataFrame({})
    count = 0
    if no_features == 'auto':
        no_features = X.shape[1] // 2
    while count < no_features:
        mutual_info_scores = [mutual_information(y, X.iloc[:, i]) for i in range(X.shape[1])]
        redundancy_scores = [redundancy(X.iloc[:, i], X) for i in range(X.shape[1])]
        scores = [mi - rd for mi, rd in zip(mutual_info_scores, redundancy_scores)]
        best_feature_index = np.argmax(scores)
        selected_features = pd.concat([selected_features, pd.DataFrame(X.iloc[:, best_feature_index])], axis=1, ignore_index=False)
        X.drop(X.columns[best_feature_index], axis=1, inplace=True)
        count += 1
    return selected_features

def AFFMI(fi, fj):
    affmi_scores = []
    for j in range(fj.shape[1]):
        scores = [mutual_information(fi.iloc[:, i], fj.iloc[:, j]) for i in range(fi.shape[1])]
        affmi_scores.append(np.mean(scores))
    return affmi_scores

def JMI(F, S, C):
    FC = joint_entropy(F, C)
    SC = joint_entropy(S, C)
    FSC = joint_entropy(np.c_[F, S], C)
    mutual_info_S_C = mutual_information(S, C)
    return FC + SC - FSC - entropy(C) + mutual_info_S_C

def AJMII(F, S, y):
    ajmi_scores = []
    for j in range(F.shape[1]):
        scores = [JMI(F.iloc[:, j], S.iloc[:, i], y) for i in range(S.shape[1])]
        ajmi_scores.append(np.mean(scores))
    return ajmi_scores

def jomic(X, y, no_features=5):
    selected_features = pd.DataFrame({})
    feature_indices = np.arange(X.shape[1])
    first_index = np.argmax(mutual_information(y, X))
    selected_features = pd.concat([selected_features, X.iloc[:, first_index]], axis=1)
    X.drop(X.columns[first_index], axis=1, inplace=True)
    for _ in range(no_features - 1):
        affmi_scores = AFFMI(selected_features, X)
        ajmi_scores = AJMII(X, selected_features, y)
        scores = np.array(ajmi_scores) - np.array(affmi_scores)
        best_feature_index = np.argmax(scores)
        selected_features = pd.concat([selected_features, X.iloc[:, best_feature_index]], axis=1)
        X.drop(X.columns[best_feature_index], axis=1, inplace=True)
    return selected_features

def intrinsic_information(X):
    _, counts = np.unique(X, return_counts=True, axis=0)
    probabilities = counts / len(X)
    return -np.sum(probabilities * np.log2(probabilities))

def gain_ratio(y, X):
    return [mutual_information(y, X.iloc[:, i]) / intrinsic_information(X.iloc[:, i]) for i in range(X.shape[1])]

def Gratio(X, y, no_features=5):
    selected_features = pd.DataFrame({})
    for _ in range(no_features):
        gain_ratios = gain_ratio(y, X)
        best_feature_index = np.argmax(gain_ratios)
        selected_features = pd.concat([selected_features, X.iloc[:, best_feature_index]], axis=1)
        X.drop(X.columns[best_feature_index], axis=1, inplace=True)
    return selected_features

def PCFS(X, y, no_features=5):
    df = pd.concat([X, y], axis=1)
    label = y.name
    sorted_columns = df.corr().sort_values(label, ascending=False).index
    sorted_df = df[sorted_columns]
    return sorted_df.drop(label, axis=1).iloc[:, :no_features]

def MIFSND(X_i, y_i, no_features=5):
    Fi = pd.DataFrame({})
    index_max_mi = np.argmax([mutual_information(y_i, X_i.iloc[:, i]) for i in range(X_i.shape[1])])
    Fi = pd.concat([Fi, X_i.iloc[:, index_max_mi]], axis=1)
    X_i.drop(X_i.columns[index_max_mi], inplace=True, axis=1)
    count = 1
    while count < no_features:
        affmi = AFFMI(Fi, X_i)
        fcmi = [mutual_information(y_i, X_i.iloc[:, i]) for i in range(X_i.shape[1])]
        best_index = np.argmax([affmi[i] - fcmi[i] for i in range(len(affmi))])
        Fi = pd.concat([Fi, X_i.iloc[:, best_index]], axis=1)
        X_i.drop(X_i.columns[best_index], inplace=True, axis=1)
        count += 1
    return Fi
