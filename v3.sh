#!/usr/bin/env bash

#Precondition: v2

#Prepare new intermediate data files with numerical features with operations
python prepare_inter_v3.py
#Train model and generate feature importance csv file for each combination train file
python main_inter_v3.py
# Combine features from intermediate files and v2 train
python prepare_v3.py
#Train model with new train
python main_v3.py
