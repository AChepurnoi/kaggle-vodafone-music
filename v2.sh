#!/usr/bin/env bash

#Precondition: v1

#Prepare new intermediate data files with categorical features combinations
python prepare_inter_v2.py
#Train model and generate feature importance csv file for each combination train file
python main_inter_v2.py
# Combine features from intermediate files and v1 train
python prepare_v2.py
#Train model with new train
python main_v2.py
