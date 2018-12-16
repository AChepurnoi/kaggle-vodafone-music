import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import itertools

from src.context import global_config


def load_test(path):
    test_lb = pd.read_pickle(path)
    return test_lb


def prepare_folds(data):
    kfold = StratifiedKFold(n_splits=global_config['folds'], random_state=42)
    folds_data = []
    for ids in kfold.split(data.id.values, data.target.values):
        train, test = ids
        folds_data.append((data.iloc[train], data.iloc[test]))

    return folds_data


def fill_na(data):
    data = data.copy()
    data.loc[:, 'sim_count'] = data.loc[:, 'sim_count'].fillna(100)
    data.loc[:, 'tp_flag'] = data.loc[:, 'tp_flag'].fillna(0)
    data.loc[:, 'service_1_flag'] = data.loc[:, 'service_1_flag'].fillna(100)
    data.loc[:, 'service_2_flag'] = data.loc[:, 'service_2_flag'].fillna(100)
    data.loc[:, 'service_3_flag'] = data.loc[:, 'service_3_flag'].fillna(100)
    data.loc[:, 'is_obl_center'] = data.loc[:, 'is_obl_center'].fillna(100)
    data.loc[:, 'inact_days_count'] = data.loc[:, 'inact_days_count'].fillna(0)
    data.loc[:, 'service_P_flag_m1'] = data.loc[:, 'service_P_flag_m1'].fillna(100)
    data.loc[:, 'service_P_flag_m2'] = data.loc[:, 'service_P_flag_m2'].fillna(100)
    data.loc[:, 'service_P_flag_m3'] = data.loc[:, 'service_P_flag_m3'].fillna(100)

    return data


def prepare_types(data, features):
    df = data.copy()
    for f, t in features.items():
        if t is 'c' and f not in ['target']:
            df.loc[:, f] = df.loc[:, f].astype(int)
        elif t is 'n':
            df.loc[:, f] = df.loc[:, f].astype(float)
    return df


def combine_cat_features(data, features, interactions=2):
    combinations = list(itertools.combinations(features, interactions))
    print("Produced {} features".format(len(combinations)))
    generated = {}
    for combination in combinations:
        f_name = "_".join(combination)
        res = data[combination[0]].astype(str)
        for feature in combination[1:]:
            res = res + "_" + data[feature].astype(str)

        generated[f_name] = res
    return pd.DataFrame(generated)


def combine_num_features(data, features, op_name, op):
    combinations = list(itertools.combinations(features, 2))
    print("Produced {} features".format(len(combinations)))
    generated = {}
    for combination in combinations:
        f_name = "_{}_".format(op_name).join(combination)
        res = op(data[combination[0]], data[combination[1]])
        generated[f_name] = res
    return pd.DataFrame(generated)


def mean_encode_test(train, test, cats):
    features = []
    for cat in cats:
        cat_name = "_".join(cat)
        feature_name = "{}_target_mean".format(cat_name)
        reduced_train = train[[*cat, 'target']]
        reduced_test = test[[*cat]]
        reduced_train_mean = reduced_train.groupby(cat).agg({'target': 'mean'}) \
            .rename(columns={'target': feature_name})
        merged = reduced_test.merge(reduced_train_mean, left_on=cat, right_index=True, how='left')[feature_name]
        features.append(merged)

    test = pd.concat([test, *features], axis=1)
    return test


def mean_encode_train_fold(folds, cats):
    resulting = []
    for j in range(len(folds)):
        print("-------> Processing fold {}".format(j))
        tf, vf = folds[j]
        m_folds = prepare_folds(tf)
        res = []
        for i in range(len(m_folds)):
            print("---> Processing inner fold {}".format(i))
            m_tf, m_vf = m_folds[i]
            mean_features = []
            for cat in cats:
                cat_name = "_".join(cat)
                feature_name = "{}_target_mean".format(cat_name)
                reduced_tf = m_tf[[*cat, 'target']]
                reduced_vf = m_vf[[*cat, 'target']]
                vf_target_mean = reduced_tf.groupby(cat) \
                    .agg({'target': 'mean'}) \
                    .rename(columns={'target': feature_name})

                reduced_merged = reduced_vf.merge(vf_target_mean, left_on=cat,
                                                  right_index=True, how='left')[feature_name]
                mean_features.append(reduced_merged)

            merged = pd.concat([m_vf, *mean_features], axis=1)
            res.append(merged)

        encoded = pd.concat(res, sort=False)
        for cat in cats:
            cat_name = "_".join(cat)
            feature_name = "{}_target_mean".format(cat_name)
            train_target_mean = tf.groupby(cat).agg({'target': 'mean'})\
                .rename(columns={'target': feature_name})
            vf = vf.merge(train_target_mean, left_on=cat, right_index=True, how='left')
        resulting.append((encoded, vf))

    return resulting
