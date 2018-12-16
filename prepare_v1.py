from src.features_list import *
from src.data import fill_na, prepare_types, combine_cat_features


train_test = pd.concat([train, test], sort=False)

train_test = fill_na(train_test)
train_test = prepare_types(train_test, all_features)

train = train_test[~train_test.target.isna()]
test = train_test[train_test.target.isna()].drop(columns=["target"])


train.to_pickle(DATA_FOLDER + '/v1/train.pkl')
test.to_pickle(DATA_FOLDER + '/v1/test.pkl')




