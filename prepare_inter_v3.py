from src.features_list import *
from src.data import fill_na, prepare_types, combine_cat_features, combine_num_features

features = set(pd.read_csv(DATA_FOLDER + '/v2/importances.csv', index_col=0).head(50).index)

train = pd.read_pickle(DATA_FOLDER + '/v2/train.pkl')
test = pd.read_pickle(DATA_FOLDER + '/v2/test.pkl')

numerical = set(train.columns[train.dtypes == float]).intersection(features)
train_test = pd.concat([train, test], sort=False)

operations = {
    'div': lambda x, y: x / y,
    'diff': lambda x, y: x - y,
    'sum': lambda x, y: x + y,
    'prod': lambda x, y: x * y
}
for op_name, op_fun in operations.items():
    gen_num_features = combine_num_features(train_test, numerical, op_name, op_fun)

    gen_num_features['target'] = train_test['target']
    gen_num_features['id'] = train_test['id']

    train = gen_num_features[~gen_num_features.target.isna()]
    test = gen_num_features[gen_num_features.target.isna()].drop(columns=["target"])

    train.to_pickle(DATA_FOLDER + '/v3/inter_{}_train.pkl'.format(op_name))
    test.to_pickle(DATA_FOLDER + '/v3/inter_{}_test.pkl'.format(op_name))
