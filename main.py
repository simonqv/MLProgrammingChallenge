import numpy as np
import pandas as pd
import random, math
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from input_stuff import check_input
from data_preprocessing import data_preprocessing
from modeling import try_modeling, try_params_adaboost, try_params_MLP, final

TRAIN_PATH = "Data/TrainOnMe-4.csv"
EVALUATION_PATH = "Data/EvaluateOnMe-4.csv"


def main():
    df = pd.read_csv(TRAIN_PATH, index_col=0)
    eval_data = pd.read_csv(EVALUATION_PATH, index_col=0)
    # check_input(df)
    preprocess, X_train, y_train, X, y = data_preprocessing(df)
    # try_modeling(df, preprocess, X_train, y_train)
    mlp = try_params_MLP(df, preprocess, X_train, y_train)
    # try_params_adaboost(df, preprocess, X_train, y_train)

    final(df, eval_data, X, y, mlp)


if __name__ == "__main__":
    main()
