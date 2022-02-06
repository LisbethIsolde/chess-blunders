import pandas as pd
from dataloader import DataLoader
from pre_processing import PreProcessing
from evaluation import Evaluation
from regression import Regression
from visualization import Visualization
import _pickle as cPickle
import gc


def create_csv_pkl():
    """
    Since the pgn files are very big we read the separately
    """

    # Load January 2015
    DataLoader.create_csv(path_databases="data", databases=["lichess_db_standard_rated_2015-1.pgn"], path_csv="data", name_csv="2015-1.csv", stockfish_analysis=True)
    dataframe = pd.read_csv("data/2015-1.csv")
    PreProcessing.convert_df(dataframe)
    PreProcessing.preprocess_gameplay(dataframe)
    print("preprocessing done")
    dataframe.to_csv("data/2015-1.csv")
    dataframe.to_pickle("data/2015-1.pkl")
    print("stored as csv and pkl")

    # Load February 2015
    DataLoader.create_csv(path_databases="data", databases=["lichess_db_standard_rated_2015-2.pgn"], path_csv="data", name_csv="2015-2.csv", stockfish_analysis=True)
    dataframe = pd.read_csv("data/2015-2.csv")
    PreProcessing.convert_df(dataframe)
    PreProcessing.preprocess_gameplay(dataframe)
    print("preprocessing done")
    dataframe.to_csv("data/2015-2.csv")
    dataframe.to_pickle("data/2015-2.pkl")
    print("stored as csv and pkl")


def read_pkl(pkl_list):
    """
    Reads the pkl files and returns a dataframe

    Parameters
    ----------
    pkl_list : list
        list of pkl files
    Return
    ------
    df : df
        pandas dataframe
    """

    # read files much faster with cPickle and disabled garbage collection, python pickle is infeasible slow
    data = []
    gc.disable()
    for pkl in pkl_list:
        output = open(pkl, "r+b")
        data.append(cPickle.load(output))
        output.close()
        print("loaded ", pkl)

    # concat files
    df = pd.concat(data)
    print("concat complete")
    gc.enable()

    return df


def create_plots(dataframe):
    """
    Creates all four plots that we used in our paper

    Parameters
    ----------
    dataframe : df
        pandas data frame
    """

    blunder_dict, moves_dict = Evaluation.create_blunder_moves_dicts(dataframe)
    Visualization.plot_blunder_heatmap(blunder_dict, moves_dict, [(800, 1400), (1400, 2100), (2100, 2700)])
    print("heatmap created")

    Visualization.plot_blunder_boxplot(blunder_dict, moves_dict, [(800, 1300), (1300, 1700), (1700, 2100), (2100, 2700)])
    print("violin plot created")

    labels = ["Elo", "Termination", "GameLength", "RemainingPieces", "Blunder1", "Blunder3",
              "Blunder9", "BlunderInf",
              "BlunderPercentagePawn", "BlunderPercentageKnight", "BlunderPercentageBishop",
              "BlunderPercentageRook", "BlunderPercentageQueen", "BlunderPercentageKing",
              "BlunderPercentageWeighted", "MovesPercentagePawn", "MovesPercentageKnight", "MovesPercentageBishop",
              "MovesPercentageRook", "MovesPercentageQueen", "MovesPercentageKing", "MovesPercentageWeighted",
              "AvgBlunderTime"]

    evaluation_module = Evaluation()
    regression_df = evaluation_module.regression(dataframe)

    # drop features to create correlation plots and compute linear regression
    drop_features = ["Termination", "RemainingPieces", "BlunderInf", "BlunderPercentageWeighted", "MovesPercentageWeighted", "AvgBlunderTime"]
    dep_val = "Elo"
    regression_df.drop(drop_features, axis=1, inplace=True)
    labels = [x for x in labels if x not in drop_features]
    labels.remove(dep_val)
    regression_module = Regression(regression_df, labels, dep_val)

    regression_module.plot_correlations()
    print("correlation plot created")
    regression_module.plot_linear_regression(test_size=0.2)
    print("regression created")


if __name__ == "__main__":
    create_csv_pkl()
    df = read_pkl(["data/2015-1.pkl", "data/2015-2.pkl"])
    create_plots(df)

