from src.features_list import *
from src.data import fill_na, prepare_types, combine_cat_features

features = list(pd.read_csv(DATA_FOLDER + '/v1/importances_filtered.csv', index_col=0).index)
train_features = [*features, 'target']

train = pd.read_pickle(DATA_FOLDER + '/v1/train.pkl')[train_features]
test = pd.read_pickle(DATA_FOLDER + '/v1/test.pkl')[features]

categorical = list(train.columns[train.dtypes == int])
train_test = pd.concat([train, test], sort=False)

interactions_features = combine_cat_features(train_test, categorical).apply(lambda x: pd.factorize(x)[0])

interactions_features['target'] = train_test['target']
interactions_features['id'] = train_test['id']

train = interactions_features[~interactions_features.target.isna()]
test = interactions_features[interactions_features.target.isna()].drop(columns=["target"])


train.to_pickle(DATA_FOLDER + '/v2/inter_2_train.pkl')
test.to_pickle(DATA_FOLDER + '/v2/inter_2_test.pkl')

# -----------------------------------------Tripple interactions-------------------------------------------------------
features = list(pd.read_csv(DATA_FOLDER + '/v1/importances_filtered.csv', index_col=0).index)
train_features = [*features, 'target']

train = pd.read_pickle(DATA_FOLDER + '/v1/train.pkl')[train_features]
test = pd.read_pickle(DATA_FOLDER + '/v1/test.pkl')[features]

categorical = list(train.columns[train.dtypes == int])
train_test = pd.concat([train, test], sort=False)

interactions_features = combine_cat_features(train_test, categorical, 3).apply(lambda x: pd.factorize(x)[0])

interactions_features['target'] = train_test['target']
interactions_features['id'] = train_test['id']

train = interactions_features[~interactions_features.target.isna()]
test = interactions_features[interactions_features.target.isna()].drop(columns=["target"])


train.to_pickle(DATA_FOLDER + '/v2/inter_3_train.pkl')
test.to_pickle(DATA_FOLDER + '/v2/inter_3_test.pkl')