import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import math
from tueplots import bundles, figsizes


class Regression:
    """
    Regression module do create correlation and linear regression plot

    Attributes
    ----------
        df : df
            pandas dataframe
        labels : list
            contains names of all features
        dep_val : str
            name of dependent value column in df
    """

    def __init__(self, df, labels, dep_val):
        self.df = df
        self.dependent_values = df[dep_val].values
        self.independent_values = df.drop(dep_val, axis=1).values
        self.labels = labels
        self.dep_val = dep_val
        self.x_ = np.arange(len(labels))

    def plot_correlations(self):
        """
        Plots correlations between Elo and each feature given in labels
        """

        # replace titles by specific hard coded strings for the specific type of plot we used in our paper
        titles = ["Game Length", "0.5 $\leq$ Blunders $\leq$ 1", "1 $<$ Blunders $\leq$ 3", "3 $<$ Blunders $\leq$ 9", "Blunders \large{\sympawn}", "Blunders \large{\symknight}", "Blunders \large{\symbishop}", "Blunders \large{\symrook}", "Blunders \large{\symqueen}", "Blunders \large{\symking}", "Moves \large{\sympawn}", "Moves \large{\symknight}", "Moves \large{\symbishop}", "Moves \large{\symrook}", "Moves \large{\symqueen}", "Moves \large{\symking}"]
        cols = 4
        plt.rcParams.update(bundles.neurips2021())
        plt.rcParams.update(figsizes.neurips2021(nrows=math.ceil(len(self.labels) / cols), ncols=cols, constrained_layout=True, tight_layout=False, height_to_width_ratio=1.0))
        plt.rc('text.latex', preamble=r'\usepackage{skak} \usepackage{times}')

        fig, ax = plt.subplots(nrows=math.ceil(len(self.labels) / cols), ncols=cols, sharey=True)
        for i, l in enumerate(self.labels):  # use just for specific labels
            X = self.df[l].values
            x_ = np.linspace(X.min(), X.max(), 100 + 1)
            y = self.dependent_values
            reg = LinearRegression(fit_intercept=True).fit(X.reshape(-1, 1), y)
            y_pred_sklearn = reg.predict(x_.reshape(-1, 1))

            # Feature Game Length should not have percentage in x-axis
            if not "GameLength" in l:
                ax[math.floor(i / cols), i % cols].xaxis.set_major_formatter(mtick.PercentFormatter(1))
                ax[math.floor(i / cols), i % cols].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}"))
            else:
                ax[math.floor(i / cols), i % cols].xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}"))
            ax[math.floor(i / cols), i % cols].plot(x_, y_pred_sklearn, color='red', alpha=0.7)
            ax[math.floor(i / cols), i % cols].scatter(self.df[l].values, self.df[self.dep_val].values, linewidth=0, alpha=0.1, s=1, rasterized=True)
            ax[math.floor(i / cols), i % cols].set_title(titles[i])  # use just for specific labels
            ax[math.floor(i / cols), i % cols].set_ylabel("Elo")
        plt.show()
        fig.savefig("plots/correlations.pdf")

    def plot_linear_regression(self, test_size):
        """
        Performs linear regression and generates the corresponding plot.

        Parameters
        ----------
        test_size : float [0,1]
            Ratio of the test-train size for the linear regression
        """

        # Standardize Data
        indep_val_standardized = preprocessing.StandardScaler().fit_transform(self.independent_values)
        # split data into training and testing sets
        independent_train, independent_test, dependent_train, dependent_test = train_test_split(indep_val_standardized, self.dependent_values.reshape(-1, 1), test_size=test_size, random_state=42)
        lin_reg = LinearRegression().fit(independent_train, dependent_train)
        test_pred = lin_reg.predict(independent_test)
        mean_squared_loss = mean_squared_error(dependent_test, test_pred)
        mean_abs_loss = mean_absolute_error(dependent_test, test_pred)
        mean_abs_per_loss = mean_absolute_percentage_error(dependent_test, test_pred)

        print("Losses: mean_squared: ", mean_squared_loss, " abs: ", mean_abs_loss, " abs_per: ", mean_abs_per_loss)

        plt.rcParams.update(bundles.neurips2021())
        plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=1, constrained_layout=True, tight_layout=False, height_to_width_ratio=0.5))
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(dependent_test, test_pred, alpha=0.5, linewidth=0, s=1, rasterized=True)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red", alpha=0.7)
        ax.set_xlim([800, 2600])
        ax.set_ylim([800, 2600])
        ax.set_title("Elo Prediction")
        ax.set_ylabel("Model Predictions")
        ax.set_xlabel("Truths")
        plt.show()
        fig.savefig("plots/regression.pdf")

