from src.features_list import *
from src.data import fill_na, prepare_types, combine_cat_features, mean_encode_train_fold, mean_encode_test, \
    prepare_folds

#
train = pd.read_pickle(DATA_FOLDER + '/v2/train.pkl')
test = pd.read_pickle(DATA_FOLDER + '/v2/test.pkl')

inter_features = list(pd.read_csv('data/v2/importances.csv', index_col=0).head(10).index)

categorical = set(train.columns[train.dtypes == int]).intersection(inter_features)


cats = [[x] for x in categorical]

folds = prepare_folds(train)

folds = mean_encode_train_fold(folds, cats)



# train_inter = pd.read_pickle('data/v4/inter_train.pkl')[inter_features]
# test_inter = pd.read_pickle('data/v4/inter_test.pkl')[inter_features]
#
#
# train = pd.concat([train, train_inter], axis=1)
# test = pd.concat([test, test_inter], axis=1)
#
# train.to_pickle('data/v4/train.pkl')
# test.to_pickle('data/v4/test.pkl')
#
#
#
#
