import numpy as np
import pandas as pd
import random, math
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

RANDOM = 6


def data_preprocessing(df):
    # print(df.dtypes)
    X = df.drop('y', axis=1)
    y = df.y

    X_train, y_train = shuffle(X, y, random_state=RANDOM)

    numerical = X.select_dtypes(include=['float64', 'int64']).columns
    categorical = X.select_dtypes(include=['object']).columns

    cat_trans = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
        ('encoder', OrdinalEncoder())
    ])

    num_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        #('pca', PCA(n_components=8)),
        ("scaler", StandardScaler()),
    ])

    preproc = ColumnTransformer(transformers=[('num', num_trans, numerical), ('cat', cat_trans, categorical)])
    return preproc, X_train, y_train, X, y
