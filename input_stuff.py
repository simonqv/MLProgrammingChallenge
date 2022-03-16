import matplotlib.pyplot as plt
import seaborn as sns


def check_input(df):
    """
    print("head\n")
    print(df.head())
    print("info\n")
    df.info()
    print("shape\n")
    print(df.shape)
    print("missing?")
    # Check if empty data exists, no empty data exists
    print(df.isnull().sum())
    print("information about columns")
    # Check information about columns.
    with pd.option_context('display.max_columns', 30):
        print(df.describe(include="all"))
"""

    print("Find relations")
    df_copy = df.copy()
    df_copy.y = df.y.astype("category").cat.codes
    # with pd.option_context('display.max_columns', 27):
    #    print(df_copy.corr())

    # plot_correlations(df)
    # correlation_heatmap(df_copy)
    # plot_categorical_values_and_labels(df_copy)


def plot_categorical_values_and_labels(df):
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

    df.et.value_counts().plot(kind="bar")
    # df.y.value_counts().plot(ax=axes[1], kind="bar")
    plt.show()
    df.y.value_counts().plot(kind="pie")
    plt.show()


def correlation_heatmap(df):
    correlations = df.corr()

    fig, ax = plt.subplots(figsize=(28, 28))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()


def plot_correlations(df):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    # numerical_attributes = ["fh","e.sa","e.n","e.d","e.a","e.su","e.f","e.h","gen","age.1","hp.y","hp.p","ho.r","s.value","le.ngeo","le.nogec","le.occlusion","le.nogeo","le.ngec","le.dg","re.ngeo","re.nogec","re.occlusion","re.nogeo","re.ngec","re.dg","et","y"]
    numerical_attributes = ["le.dg", "re.dg"]
    labels = df.y.unique()
    colors = ['blue', 'red']
    print("hej")
    for ax_i, i in enumerate(numerical_attributes):
        for ax_j, j in enumerate(numerical_attributes):
            x = str(i)
            y = str(j)
            for k, label in enumerate(labels):
                df[df.y == label].plot(kind='scatter', x=x, y=y, ax=axes[ax_i][ax_j],
                                       c=colors[k], )

            axes[ax_i][ax_j].set_xlabel(x)
            axes[ax_i][ax_j].set_ylabel(y)
        print(ax_i)
    print("here???")
    plt.show()
    print("done")
