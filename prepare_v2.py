from src.features_list import *
from src.data import fill_na, prepare_types, combine_cat_features

features = list(pd.read_csv(DATA_FOLDER + '/v1/importances_filtered.csv', index_col=0).index)
train_features = [*features, 'target']

train = pd.read_pickle(DATA_FOLDER + '/v1/train.pkl')[train_features]
test = pd.read_pickle(DATA_FOLDER + '/v1/test.pkl')[features]

inter_features_2 = list(pd.read_csv(DATA_FOLDER + '/v2/inter_2_importances.csv', index_col=0).index)
inter_features_2.remove('id')
train_inter_2 = pd.read_pickle(DATA_FOLDER + '/v2/inter_2_train.pkl')[inter_features_2]
test_inter_2 = pd.read_pickle(DATA_FOLDER + '/v2/inter_2_test.pkl')[inter_features_2]

inter_features_3 = list(pd.read_csv(DATA_FOLDER + '/v2/inter_3_importances.csv', index_col=0).index)
inter_features_3.remove('id')
train_inter_3 = pd.read_pickle(DATA_FOLDER + '/v2/inter_3_train.pkl')[inter_features_3]
test_inter_3 = pd.read_pickle(DATA_FOLDER + '/v2/inter_3_test.pkl')[inter_features_3]


train = pd.concat([train, train_inter_2, train_inter_3], axis=1)
test = pd.concat([test, test_inter_2, test_inter_3], axis=1)

train.to_pickle(DATA_FOLDER + '/v2/train.pkl')
test.to_pickle(DATA_FOLDER + '/v2/test.pkl')




