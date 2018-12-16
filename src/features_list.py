import pandas as pd
import numpy as np
import itertools

from src.context import DATA_FOLDER

train = pd.read_csv(DATA_FOLDER + '/train_music.csv')
test = pd.read_csv(DATA_FOLDER + '/test_music.csv')

device_features = {
    'id': 'n',
    'device_type': 'c',
    'manufacturer_category': 'c',
    'os_category': 'c',
    'sim_count': 'c'
}

general_features = {
    'tp_flag': 'c',
    'lt': 'n',
    'block_flag': 'c',
    'days_exp': 'n',
    'service_1_flag': 'c',
    'service_1_count': 'n',
    'service_2_flag': 'c',
    'service_3_flag': 'c',
    'is_obl_center': 'c',
    'is_my_vf': 'c'
}

user_activity_features = {
    'balance_sum': 'n',
    'paym_last_days': 'n',
    'inact_days_count': 'c'
}

dynamics_features_generic = {
    'service_P_flag': 'c',
    'block_all_dur': 'n',
    'block_count': 'n',
    'all_cost': 'n',
    'all_home_clc': 'n',
    'all_roam_cost': 'n',
    'sms_cost': 'n',
    'sms_roam_cost': 'n',
    'content_cost': 'n',
    'abon_cost': 'n',
    'abon_part': 'n',
    'act_days_count': 'n',
    'com_num_cost': 'n',
    'conn_com_cost': 'n',
    'paym_el_count': 'n',
    'paym_el_sum': 'n',
    'paym_sum': 'n',
    'pay_in_P2P_cost': 'n',
    'pay_out_P2P_cost': 'n',
    'paym_count': 'n'
}
dynamics_features_m1 = {'{}_m1'.format(x): y for x, y in dynamics_features_generic.items()}
dynamics_features_m2 = {'{}_m2'.format(x): y for x, y in dynamics_features_generic.items()}
dynamics_features_m3 = {'{}_m3'.format(x): y for x, y in dynamics_features_generic.items()}

all_features = {
    **device_features,
    **general_features,
    **user_activity_features,
    **dynamics_features_m1,
    **dynamics_features_m2,
    **dynamics_features_m3
}

other_features = set(train.columns).difference(set(all_features.keys()))

all_features = {
    **all_features,
    **{f: 'n' for f in other_features},
    'target': 'c'
}

# del all_features['target']
