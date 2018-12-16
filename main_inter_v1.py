from src.features_list import *
from src.data import fill_na, prepare_types, combine_cat_features, prepare_folds, load_test
from src.train import train_folds, evaluate
from src.utils import prepare_submission

config = {
    'random_state': 42,
    'num_iterations': 5000,
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 12,
    'max_depth': 4,
    'min_data_in_leaf': 500,
    'reg_alpha': 5,  # L1 regularization term on weights
    'reg_lambda': 50,
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'scale_pos_weight': 9,  # because training data is unbalanced
    'verbose': -1
}

train = pd.read_pickle(DATA_FOLDER + '/v1/train.pkl')
folds = prepare_folds(train)

models, result = train_folds(folds, config)

test = load_test(DATA_FOLDER + '/v1/test.pkl')
test_target = evaluate(models, test)

print("AUC: %.4f, F1: %.4f" % (result['auc'], result['f1']))

importance = result['importances'].groupby(['feature']) \
    .agg({'importance': 'mean'}) \
    .sort_values(by="importance", ascending=False)

importance.to_csv(DATA_FOLDER + "/v1/importances.csv")

prepare_submission(test_target, "v1_AUC_%.4f_F1_%.4f" % (result['auc'], result['f1']))
