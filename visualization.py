import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tueplots import figsizes
import matplotlib.ticker as mtick
from evaluation import Evaluation


class Visualization:
    """
    Visualization module to create violin plot and heatmap
    """

    @staticmethod
    def plot_blunder_boxplot(blunders_dict, moves_dict, elos):
        """
        Creating a box/violin plot of all blunders per piece at different Elo ranges
        Writing boxplot.pdf to directory containing plots

        Parameters
        ----------
        blunders_dict : dict
            Nested dictionary, where Elos are main keys. Per Elo, the value is a second
            dictionary containing all blunder per chess piece

        moves_dict : dict
            Nested dictionary, where Elos are main keys. Per Elo, the value is a second
            dictionary containing all moves per chess piece

        elos : [(Int, Int)]
            List of Elo ranges to account for in the boxplox
        """

        plt.rcParams.update(bundles.neurips2021(family="sans-serif"))
        plt.rcParams.update(figsizes.neurips2021(nrows=2, ncols=2))
        plt.rc('text.latex', preamble=r'\usepackage{skak} \usepackage{times}')

        fig, axes = plt.subplots(2, 2)

        for i, ax in enumerate(axes.flat):
            elo_range = elos[i]

            blunders_elo = blunders_dict[elo_range[0]]
            moves_elo = moves_dict[elo_range[0]]

            for elo in range(elo_range[0] + 100, elo_range[1], 100):
                blunders_elo = Evaluation.merge_dicts_of_lists(blunders_elo, blunders_dict[elo])
                moves_elo = Evaluation.merge_dicts_of_lists(moves_elo, moves_dict[elo])

            blunders_p = [x[1] for x in blunders_elo["P"]]
            blunders_n = [x[1] for x in blunders_elo["N"]]
            blunders_b = [x[1] for x in blunders_elo["B"]]
            blunders_r = [x[1] for x in blunders_elo["R"]]
            blunders_q = [x[1] for x in blunders_elo["Q"]]
            blunders_k = [x[1] for x in blunders_elo["K"]]

            moves_p = [x[1] for x in moves_elo["P"]]
            moves_n = [x[1] for x in moves_elo["N"]]
            moves_b = [x[1] for x in moves_elo["B"]]
            moves_r = [x[1] for x in moves_elo["R"]]
            moves_q = [x[1] for x in moves_elo["Q"]]
            moves_k = [x[1] for x in moves_elo["K"]]

            total_blunders = np.array(
                [len(blunders_p), len(blunders_n), len(blunders_b), len(blunders_r), len(blunders_q), len(blunders_k)])
            total_moves = np.array([len(moves_p), len(moves_n), len(moves_b), len(moves_r), len(moves_q), len(moves_k)])
            ax2 = ax.twinx()
            ax.violinplot([blunders_p, blunders_n, blunders_b, blunders_r, blunders_q, blunders_k], showmedians=True)
            # blunder perc
            letter = chr(i + 97)
            ax2.bar(range(1, 7), total_blunders / total_moves, alpha=0.2)
            ax.set_ylim(bottom=0)
            ax.set_title(f"({letter}) {elo_range[0]} $\leq$ Elo $<$ {elo_range[1]}")
            if i % 2 == 0:
                ax.set_ylabel("Weight of Blunder")
            elif i % 2 == 1:
                ax2.set_ylabel("Rate of Blunders to Moves")

            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0f}'))
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_xticks(range(1, 7))
            ax.set_xticklabels(["\large{\sympawn}", "\large{\symknight}", "\large{\symbishop}", "\large{\symrook}",
                                "\large{\symqueen}", "\large{\symking}"])
        fig.savefig("./plots/boxplot_single.pdf")
        plt.show()

    @staticmethod
    def plot_blunder_heatmap(blunders_dict, moves_dict, elos):
        """
        Creating a heatmap of blunder ratio per square on the chess
        Writing heatmap.pdf to directory containing plots

        Parameters
        ----------
        blunder_dict : dict
            Nested dictionary, where Elos are main keys. Per Elo, the value is a second
            dictionary containing all blunder per chess piece

        moves_dict : dict
            Nested dictionary, where Elos are main keys. Per Elo, the value is a second
            dictionary containing all moves per chess piece

        elos : [(Int, Int)]
            List of Elo ranges to account for in the heatmap
        """

        plt.rcParams.update(bundles.neurips2021())
        plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=3, constrained_layout=True, height_to_width_ratio=1))

        fig, axes = plt.subplots(1, 3)

        for i, ax in enumerate(axes.flat):
            elo_range = elos[i]

            # Make a 8x8 grid
            nrows, ncols = 8, 8
            # initialize arrays for heatmap
            num_blunders = np.zeros(nrows * ncols)
            weighted_blunders = np.zeros(nrows * ncols)
            num_moves = np.zeros(nrows * ncols)
            blunders_elo = blunders_dict[elo_range[0]]
            moves_elo = moves_dict[elo_range[0]]

            # combine all blunders for given elo range
            for elo in range(elo_range[0] + 100, elo_range[1], 100):
                blunders_elo = Evaluation.merge_dicts_of_lists(blunders_elo, blunders_dict[elo])
                moves_elo = Evaluation.merge_dicts_of_lists(moves_elo, moves_dict[elo])

            # fill array for heatmap with blunders, based on the square the blunder happened on
            for piece in blunders_elo:
                for field, eval_diff, player, _, _, _ in blunders_elo[piece]:
                    square = field
                    if "O-O-O" == field:  # account for castling (set kings position)
                        square = "c1" if player == "w" else "c8"
                    if "O-O" == field:  # account for castling (set kings position)
                        square = "g1" if player == "w" else "g8"

                    idx = Evaluation.square_to_index(square)
                    num_blunders[idx] += 1
                    weighted_blunders[idx] += eval_diff

            # fill array for heatmap with moves, based on the square the blunder happened on
            for piece in moves_elo:

                for field, player, _, _, _ in moves_elo[piece]:
                    square = field
                    if "O-O-O" == field:  # account for castling (set kings position)
                        square = "c1" if player == "w" else "c8"
                    if "O-O" == field:  # account for castling (set kings position)
                        square = "g1" if player == "w" else "g8"

                    idx = Evaluation.square_to_index(square)
                    num_moves[idx] += 1

            print(sum(num_blunders), " blunders found")
            print(sum(num_moves), " moves found")
            print("Blunder Move ratio: ", sum(num_blunders) / sum(num_moves))

            heatmap_data = np.divide(num_blunders, num_moves, np.zeros_like(num_blunders), where=num_moves != 0) * 100

            # Reshape things into a 8x8 grid.
            heatmap_data = heatmap_data.reshape((nrows, ncols))

            # define labels of chess board
            row_labels = list(map(str, range(nrows, 0, -1)))
            col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

            heatmap = ax.matshow(heatmap_data, cmap="magma", vmin=0, vmax=11)  # max in our dataset is approx. 10%
            letter = chr(i + 97)
            ax.set_title(f"({letter}) {elo_range[0]} $\leq$ Elo $<$ {elo_range[1]}")

            ax.tick_params(top=False, bottom=True,
                           labeltop=False, labelbottom=True)
            ax.set_xticks(range(ncols))
            ax.set_yticks(range(nrows))
            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(row_labels)

        fig.colorbar(heatmap, ax=axes.ravel().tolist(), fraction=0.048, pad=0.04,
                     label='\% of Moves that were Blunders')
        fig.savefig("./plots/heatmap.pdf")
        plt.show()