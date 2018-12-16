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
import catboost
from src.context import global_config
from src.data import prepare_folds

def train_cat(train, val, config):
    train_x = train.drop(columns=['target'])
    train_y = train.target

    test_x = val.drop(columns=['target'])
    test_y = val.target

    cols = list(train_x.columns)
    categorical_features = [cols.index(col_name)for col_name in \
                            list(train_x.dtypes[(train_x.dtypes == int)].index)]

    clf = catboost.CatBoostClassifier(**config, loss_function='Logloss')
    clf.fit(train_x, train_y, eval_set=(test_x, test_y), cat_features=categorical_features)

    predicted = clf.predict_proba(test_x)[:,1]
    score = roc_auc_score(test_y, predicted)
    prec = precision_score(test_y, (predicted > 0.5).astype(int))
    f1 = f1_score(test_y, (predicted > 0.5).astype(int))
    print("Confusion matrix:")
    print(confusion_matrix(test_y, (predicted > 0.5).astype(int)))
    print("Report:")
    print(classification_report(test_y, (predicted > 0.5).astype(int)))

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = cols
    fold_importance_df["importance"] = clf.feature_importances_

    return {'score': score, 'model': clf, 'prec': prec, 'f1': f1, 'fold_importance': fold_importance_df}


def train_lgbm(train, val, config):
    train_x = train.drop(columns=['target']).fillna(-9999)
    train_y = train.target

    test_x = val.drop(columns=['target']).fillna(-9999)
    test_y = val.target

    cols = list(train_x.columns)

    xgtrain = lgb.Dataset(train_x, label=train_y)

    xgvalid = lgb.Dataset(test_x, label=test_y)

    clf = lgb.train(config, xgtrain,
                    valid_sets=[xgtrain, xgvalid],
                    valid_names=['train', 'valid'],
                    early_stopping_rounds=global_config['early_stop'],
                    verbose_eval=50)

    predicted = clf.predict(test_x)
    score = roc_auc_score(test_y, predicted)
    prec = precision_score(test_y, (predicted > 0.5).astype(int))
    f1 = f1_score(test_y, (predicted > 0.5).astype(int))
    print("Confusion matrix:")
    print(confusion_matrix(test_y, (predicted > 0.5).astype(int)))
    print("Report:")
    print(classification_report(test_y, (predicted > 0.5).astype(int)))

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = cols
    fold_importance_df["importance"] = clf.feature_importance()

    return {'score': score, 'model': clf, 'prec': prec, 'f1': f1, 'fold_importance': fold_importance_df}


def evaluate(models, data, model_type='lgbm'):
    predictions = []
    for model in models:
        if model_type is "lgbm":
            predict = model.predict(data)
        elif model_type is 'catboost':
            predict = model.predict_proba(data)[:, 1]
        else:
            raise NotImplementedError()
        predictions.append(predict)
    return np.mean(predictions, axis=0)


def train_folds(folds, config, model='lgbm'):
    if model is 'lgbm':
        train_fun = train_lgbm
    elif model is 'catboost':
        train_fun = train_cat
    else:
        raise NotImplementedError()

    models = []
    auc = []
    f1 = []
    importances = []
    for n, fold in enumerate(folds):
        train_f, val_f = fold
        print("Training on %s" % str(train_f.shape))
        result = train_fun(train_f, val_f, config)

        importance = result['fold_importance']
        importance['fold'] = n
        importances.append(importance)

        models.append(result['model'])
        auc.append(result['score'])
        f1.append(result['f1'])
        print("Fold %s: %.4f, F1: %.4f, Precision: %.4f" % (n, result['score'], result['f1'], result['prec']))
    return models, {
        'auc': np.mean(auc),
        'importances': pd.concat(importances, sort=False),
        'f1': np.mean(f1)
    }


def top_k_by_importance(train, config, k=50):
    folds = prepare_folds(train)
    models, results = train_folds(folds, config)
    importances = results['importances']
    return list(importances.head(k).index)
