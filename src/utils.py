import pandas as pd

from src.context import DATA_FOLDER


def prepare_submission(predictions, score="Unknown"):
    sub = pd.read_csv(DATA_FOLDER + '/sample_submission_music.csv')
    sub.prediction = predictions
    sub.to_csv('submissions/s-%s.csv' % score, index=False)
