#!/usr/bin/env bash

#Prepare train and test dataframes from raw data
python prepare_v1.py
#Train model and generate feature importance csv file
python main_inter_v1.py
#Train model with top K features
python main_v1.py
