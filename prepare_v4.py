from src.features_list import *
from src.data import fill_na, prepare_types, combine_cat_features, aggregates_features
import tqdm
import itertools

features = set(pd.read_csv(DATA_FOLDER + '/v2/importances.csv', index_col=0).head(25).index)

train = pd.read_pickle(DATA_FOLDER + '/v2/train.pkl')
test = pd.read_pickle(DATA_FOLDER + '/v2/test.pkl')

numerical = set(train.columns[train.dtypes == float]).intersection(features)
categorical = set(train.columns[train.dtypes == int]).intersection(features)

transformations = {x: ['mean', 'median', 'sum', 'std', 'max', 'min'] for x in numerical}

receipts = {x: transformations for x in categorical}
train_test = pd.concat([train, test], sort=False)


def statistics_features(train_test, receipts):
    f = []
    for k, v in tqdm.tqdm(receipts.items()):
        new_features = aggregates_features(train_test, ([k], v))
        f.append(new_features)
    df = pd.concat(f, axis=1)
    return df


stat_features = statistics_features(train_test, receipts)
train_test = pd.concat([train_test, stat_features], axis=1)

train = train_test[~train_test.target.isna()]
test = train_test[train_test.target.isna()].drop(columns=["target"])

train.to_pickle(DATA_FOLDER + '/v4/train.pkl')
test.to_pickle(DATA_FOLDER + '/v4/test.pkl')
