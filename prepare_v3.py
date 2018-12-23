from src.features_list import *
from src.data import fill_na, prepare_types, combine_cat_features, filter_features

train = pd.read_pickle(DATA_FOLDER + '/v2/train.pkl')
test = pd.read_pickle(DATA_FOLDER + '/v2/test.pkl')

train_list = [train]
test_list = [test]

for op, k in {'diff': 500, 'div': 500, 'sum': 500, 'prod': 500}.items():
    inter_features = list(pd.read_csv('data/v3/inter_{}_importances.csv'.format(op), index_col=0).head(k).index)
    inter_features = [x for x in inter_features if 'id' != x]
    print("Features: {}".format(inter_features))

    train_inter = pd.read_pickle(DATA_FOLDER + '/v3/inter_{}_train.pkl'.format(op))[inter_features]
    test_inter = pd.read_pickle(DATA_FOLDER + '/v3/inter_{}_test.pkl'.format(op))[inter_features]

    train_list.append(train_inter)
    test_list.append(test_inter)


train = pd.concat(train_list, axis=1)
test = pd.concat(test_list, axis=1)

# train_test = pd.concat([train, test], sort=False)
#
# train_test = filter_features(train_test)
#
# train = train_test[~train_test.target.isna()]
# test = train_test[train_test.target.isna()].drop(columns=["target"])

train.to_pickle(DATA_FOLDER + '/v3/train.pkl')
test.to_pickle(DATA_FOLDER + '/v3/test.pkl')
