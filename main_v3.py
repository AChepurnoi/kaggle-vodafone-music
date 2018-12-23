from src.features_list import *
from src.data import fill_na, prepare_types, combine_cat_features, prepare_folds, load_test
from src.train import train_folds, evaluate
from src.utils import prepare_submission

config = {
    'random_state': 42,
    'n_estimators': 4000,
    'learning_rate': 0.03,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 30,
    'max_depth': 7,
    'min_data_in_leaf': 500,
    'reg_alpha': .1,  # L1 regularization term on weights
    'reg_lambda': .1,
    'subsample': .9,
    'min_split_gain': .01,
    'min_child_weight': 2,
    'colsample_bytree': .9,  # Subsample ratio of columns when constructing each tree.
    'scale_pos_weight': 9,  # because training data is unbalanced
    'verbose': -1
}

features = list(pd.read_csv(DATA_FOLDER + '/v3/importances.csv', index_col=0).head(800).index)
train_features = [*features, "target"]


train = pd.read_pickle(DATA_FOLDER + '/v3/train.pkl')[train_features]
folds = prepare_folds(train)

models, result = train_folds(folds, config)

test = load_test(DATA_FOLDER + '/v3/test.pkl')[features]
test_target = evaluate(models, test)

print("AUC: %.4f, F1: %.4f" % (result['auc'], result['f1']))

importance = result['importances'].groupby(['feature']) \
    .agg({'importance': 'mean'}) \
    .sort_values(by="importance", ascending=False)

importance.to_csv(DATA_FOLDER + "/v3/importances.csv")

prepare_submission(test_target, "v3_AUC_%.4f_F1_%.4f" % (result['auc'], result['f1']))
