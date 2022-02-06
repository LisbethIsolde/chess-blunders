import pandas as pd
from tqdm import tqdm
import math
import copy
 

class Evaluation:
    """
    Evaluation module that evaluates the preprocessed data frame further into forms such that different plots can
    be generated.
    """

    @staticmethod
    def create_blunder_moves_dicts(df, blunder_range=(2, 9)):
        """
        Creating two dictionaries, one containing all moves and one containing all blunders
        in the given dataframe. 

        Parameters
        ----------
        df : pd.df
            dataframe containing data on chess games

        blunder_range : (int, int)
            tuples of upper and lower value of what should be considered a blunder

        Return
        ------
        blunder_dict : dict
            Nested dictionary, where Elos are main keys. Per Elo, the value is a second 
            dictionary containing all blunder per chess piece
        
        moves_dict : dict 
            Nested dictionary, where Elos are main keys. Per Elo, the value is a second 
            dictionary containing all moves per chess piece
        """

        pieces_dict = {"P": [], "N": [], "B": [], "R": [], "Q": [], "K": []}
        blunders_by_elo = {k: copy.deepcopy(pieces_dict) for k in range(800, 2700, 100)}
        moves_by_elo = {k: copy.deepcopy(pieces_dict) for k in range(800, 2700, 100)}

        # create blunder dict
        for index, row in tqdm(df.iterrows()):
            # ignore games without blunders
            if len(row.Blunders) == 0:
                continue
            # ignore games where skill level of the players differ too much
            if abs(row.WhiteElo - row.BlackElo) > 100:
                continue
            match_elo = min(row.WhiteElo, row.BlackElo)
            # ignoring too bad or too good player (too little data available)
            if match_elo < 800 or match_elo > 2600:
                continue
            # ignore games that last less than 15 moves
            if len(row.Gameplay) < 15:
                continue

            result = row.Result
            # floor match_elo to nearest 100
            match_elo = math.floor(match_elo / 100) * 100

            ######################
            # start blunder dict #
            ######################

            for blunder in row.Blunders:
                num_move, player, move, ann, eval_diff = blunder

                # ignore too small or too large blunders
                if eval_diff < blunder_range[0] or eval_diff > blunder_range[1]:
                    continue

                player_num = 0 if player == "w" else 1
                remaining_pieces = row.Gameplay[num_move - 1][player_num][3]
                # account for promotions,
                # they are marked with uppersize char at the end of move
                if move[-1].isupper() and move[-1] != "O":
                    move = move[:-1]
                square = move[-2:]
                
                # decide which piece did the blunder
                if move[0].isupper():
                    piece = move[0]
                    # account for castling and set moving piece to king
                    if piece == "O":
                        piece = "K"
                        if "O-O-O" in move:
                            square = "O-O-O"
                        elif "O-O" in move:
                            square = "O-O"
                else:
                    piece = "P"
                # append data to blunder dict at correct keys
                blunders_by_elo[match_elo][piece].append([square, eval_diff, player, result, num_move, remaining_pieces])

            ###################
            # start move_dict #
            ###################

            for i, move in enumerate(row.Gameplay):
                if len(move) < 2:  # last move before game ends
                    break

                for j, half_move in enumerate(move):
                    # account for promotions,
                    # they are marked with uppercase char at the end
                    if half_move[0][-1].isupper() and half_move[0][-1] != "O":
                        half_move[0] = half_move[0][:-1]
                    square = half_move[0][-2:]

                    # find piece which did the move
                    if half_move[0][0].isupper():
                        piece = half_move[0][0]
                        # account for castling, and set moving piece to king
                        if piece == "O":
                            piece = "K"
                            if "O-O-O" in half_move:
                                square = "O-O-O"
                            elif "O-O" in half_move:
                                square = "O-O"
                    else:
                        piece = "P"
                    player = "w" if j == 0 else "b"
                    num_move = i + 1
                    remaining_pieces = half_move[3]

                    # append data to moves dict at correct keys
                    moves_by_elo[match_elo][piece].append([square, player, result, num_move, remaining_pieces])

        return blunders_by_elo, moves_by_elo

    @staticmethod 
    def merge_dicts_of_lists(dict1, dict2):
        """
        merge to dictionaries with the same keys, by appending their list values

        Parameters
        ----------
        dict1 : dict
        dict2 : dict

        Return
        ------
        dict : dict
        """
        return {key: value + dict2[key] for key, value in dict1.items()}

    @staticmethod
    def square_to_index(square):
        """
        calculate the correct index in the array (len=64) based on a given square (e.g. e4)

        Parameters
        ----------
        square : char, int

        Return
        ------
        index : int
        """
        letter = (ord(square[0]) - 97)  # ASCI code of a = 97
        number = 8 - int(square[1])
        index = (8 * number) + letter
        return index



    def regression(self, dataframe):
        """
        Creates a dataframe which is later used for the regression module

        Parameters
        ----------
        dataframe : df
            pandas dataframe

        Return
        ------
        all_elos : df
            pandas dataframe containing features of blunders
        """

        # create new dataframe for linear regression
        data = []

        for index, row in tqdm(dataframe.iterrows()):
            match_elo = min(row.WhiteElo, row.BlackElo)
            if match_elo < 800 or match_elo > 2600 or abs(row.WhiteElo - row.BlackElo) > 100:
                continue

            # check if match contains blunders
            if not list(filter(lambda x: x[1] == "w" and x[4] >= 1, row.Blunders)):  # not enough white blunders
                continue
            if not list(filter(lambda x: x[1] == "b" and x[4] >= 1, row.Blunders)):  # not enough black blunders
                continue

            # we discovered a strange data
            gameplay = row.Gameplay
            incomplete_data = False
            for m in gameplay:
                if len(m) < 2 and gameplay.index(m) + 1 != len(gameplay):
                    incomplete_data = True
                    break
            if incomplete_data:
                continue

            white_elo = row.WhiteElo
            black_elo = row.BlackElo
            game_length = len(row.Gameplay)

            if game_length < 15:
                continue

            termination = 0 if row.Termination == "Normal" else 1
            remaining_pieces_white = row.Gameplay[-1][0][3]
            remaining_pieces_black = row.Gameplay[-1][1][3] if len(row.Gameplay[-1]) == 2 else row.Gameplay[-2][1][3]
            blunder1_white = len(list(filter(lambda x: x[1] == "w" and x[4] <= 1, row.Blunders))) / game_length
            blunder3_white = len(list(filter(lambda x: x[1] == "w" and 1 < x[4] <= 3, row.Blunders))) / game_length
            blunder9_white = len(list(filter(lambda x: x[1] == "w" and 3 < x[4] <= 9, row.Blunders))) / game_length
            blunderInf_white = len(list(filter(lambda x: x[1] == "w" and 9 < x[4] < float("inf"), row.Blunders))) / game_length
            blunder1_black = len(list(filter(lambda x: x[1] == "b" and x[4] <= 1, row.Blunders))) / game_length
            blunder3_black = len(list(filter(lambda x: x[1] == "b" and 1 < x[4] <= 3, row.Blunders))) / game_length
            blunder9_black = len(list(filter(lambda x: x[1] == "b" and 3 < x[4] <= 9, row.Blunders))) / game_length
            blunderInf_black = len(list(filter(lambda x: x[1] == "b" and 9 < x[4] < float("inf"), row.Blunders))) / game_length
            blunders_prc_p_white = self.blunder_percentage_piece(row.Blunders, "P", "w")
            blunders_prc_n_white = self.blunder_percentage_piece(row.Blunders, "N", "w")
            blunders_prc_b_white = self.blunder_percentage_piece(row.Blunders, "B", "w")
            blunders_prc_r_white = self.blunder_percentage_piece(row.Blunders, "R", "w")
            blunders_prc_q_white = self.blunder_percentage_piece(row.Blunders, "Q", "w")
            blunders_prc_k_white = self.blunder_percentage_piece(row.Blunders, "K", "w")
            blunders_prc_p_black = self.blunder_percentage_piece(row.Blunders, "P", "b")
            blunders_prc_n_black = self.blunder_percentage_piece(row.Blunders, "N", "b")
            blunders_prc_b_black = self.blunder_percentage_piece(row.Blunders, "B", "b")
            blunders_prc_r_black = self.blunder_percentage_piece(row.Blunders, "R", "b")
            blunders_prc_q_black = self.blunder_percentage_piece(row.Blunders, "Q", "b")
            blunders_prc_k_black = self.blunder_percentage_piece(row.Blunders, "K", "b")
            blunders_prc_weighted_white = blunders_prc_p_white + blunders_prc_n_white * 3 + blunders_prc_b_white * 3 + blunders_prc_r_white * 5 + blunders_prc_q_white * 9
            blunders_prc_weighted_black = blunders_prc_p_black + blunders_prc_n_black * 3 + blunders_prc_b_black * 3 + blunders_prc_r_black * 5 + blunders_prc_q_black * 9
            moves_prc_p_white = self.moves_percentage_piece(row.Gameplay, "P", 0)
            moves_prc_n_white = self.moves_percentage_piece(row.Gameplay, "N", 0)
            moves_prc_b_white = self.moves_percentage_piece(row.Gameplay, "B", 0)
            moves_prc_r_white = self.moves_percentage_piece(row.Gameplay, "R", 0)
            moves_prc_q_white = self.moves_percentage_piece(row.Gameplay, "Q", 0)
            moves_prc_k_white = self.moves_percentage_piece(row.Gameplay, "K", 0)
            moves_prc_p_black = self.moves_percentage_piece(row.Gameplay, "P", 1)
            moves_prc_n_black = self.moves_percentage_piece(row.Gameplay, "N", 1)
            moves_prc_b_black = self.moves_percentage_piece(row.Gameplay, "B", 1)
            moves_prc_r_black = self.moves_percentage_piece(row.Gameplay, "R", 1)
            moves_prc_q_black = self.moves_percentage_piece(row.Gameplay, "Q", 1)
            moves_prc_k_black = self.moves_percentage_piece(row.Gameplay, "K", 1)
            moves_prc_weighted_white = moves_prc_p_white + moves_prc_n_white * 3 + moves_prc_b_white * 3 + moves_prc_r_white * 5 + moves_prc_q_white * 9
            moves_prc_weighted_black = moves_prc_p_black + moves_prc_n_black * 3 + moves_prc_b_black * 3 + moves_prc_r_black * 5 + moves_prc_q_black * 9
            avg_blunder_time_white = self.avg_blunder_time(row.Gameplay, row.Blunders, "w")
            avg_blunder_time_black = self.avg_blunder_time(row.Gameplay, row.Blunders, "b")

            data.append([0, white_elo, termination, game_length, remaining_pieces_white, blunder1_white, blunder3_white, blunder9_white,
                       blunderInf_white, blunders_prc_p_white, blunders_prc_n_white, blunders_prc_b_white, blunders_prc_r_white, blunders_prc_q_white, blunders_prc_k_white, blunders_prc_weighted_white,
                       moves_prc_p_white, moves_prc_n_white, moves_prc_b_white, moves_prc_r_white, moves_prc_q_white, moves_prc_k_white, moves_prc_weighted_white,
                       avg_blunder_time_white])

            data.append([1, black_elo, termination, game_length, remaining_pieces_black, blunder1_black, blunder3_black, blunder9_black,
                       blunderInf_black, blunders_prc_p_black, blunders_prc_n_black, blunders_prc_b_black,
                       blunders_prc_r_black, blunders_prc_q_black, blunders_prc_k_black, blunders_prc_weighted_black,
                       moves_prc_p_black, moves_prc_n_black, moves_prc_b_black, moves_prc_r_black, moves_prc_q_black,
                       moves_prc_k_black, moves_prc_weighted_black,
                       avg_blunder_time_black])

        data_header = ["Color", "Elo", "Termination", "GameLength", "RemainingPieces", "Blunder1", "Blunder3",
                     "Blunder9", "BlunderInf",
                     "BlunderPercentagePawn", "BlunderPercentageKnight", "BlunderPercentageBishop",
                     "BlunderPercentageRook", "BlunderPercentageQueen", "BlunderPercentageKing",
                     "BlunderPercentageWeighted", "MovesPercentagePawn", "MovesPercentageKnight", "MovesPercentageBishop",
                     "MovesPercentageRook", "MovesPercentageQueen", "MovesPercentageKing", "MovesPercentageWeighted",
                     "AvgBlunderTime"]
        # return df with all elos
        all_elos = pd.DataFrame.from_records(data, columns=data_header)

        return all_elos

    def blunder_percentage_piece(self, blunders, piece, player):
        """
        Calculates the percentage a given player blunders with the given piece.

        Parameters
        ----------
        blunders : list
            List containing all blunders of match.

        piece : char
            Chess piece \in {P, N, B, R, Q, K}

        player : char
            Player \in {w, b}

        Return
        ------
        float : float [0,1]
            percentage
        """
        blunders = list(filter(lambda x: x[1] == player and abs(x[4]) < float("inf"), blunders))
        blunders_total = len(blunders)
        if blunders_total == 0:
            return 0
        if piece == "P":
            return len(list(filter(lambda x: x[2][0].islower() and x[1] == player, blunders))) / blunders_total
        if piece == "K":
            return len(list(filter(lambda x: (x[2][0] == piece or x[2][0] == "O") and x[1] == player,
                                   blunders))) / blunders_total
        return len(list(filter(lambda x: x[2][0] == piece and x[1] == player, blunders))) / blunders_total

    def moves_percentage_piece(self, game, piece, player):
        """
        Calculates the percentage a given player moves with the given piece.

        Parameters
        ----------
        game : list
            List containing all moves of the match.

        piece : char
            Chess piece \in {P, N, B, R, Q, K}

        player : char
            Player \in {w, b}

        Return
        ------
        float : float [0,1]
            percentage
        """
        if player == 1 and len(game[-1]) < 2:
            moves_total = len(game) - 1
            gameplay = game[:-1]
        else:
            moves_total = len(game)
            gameplay = game
        if piece == "P":
            return len(list(filter(lambda x: x[player][0][0].islower(), gameplay))) / moves_total
        if piece == "K":
            return len(list(filter(lambda x: x[player][0][0] == piece or x[player][0][0] == "O", gameplay))) / moves_total

        return len(list(filter(lambda x: x[player][0][0] == piece, gameplay))) / moves_total

    def avg_blunder_time(self, gameplay, blunders, player):
        """
        Calculates how many pieces the given player had left on the board when he blunders.

        Parameters
        ----------
        gameplay : list
            List containing all moves of the match.

        blunders : list
            List containing all blunders of the match.

        player : char
            Player \in {w, b}

        Return
        ------
        float : float [1,16]
        """
        blunders = list(filter(lambda x: x[1] == player and 1 <= abs(x[4]) < float("inf"), blunders))
        blunders_total = len(blunders)
        sum_remaining_pieces = 0
        player_id = 0 if player == "w" else 0
        for blunder in blunders:
            move = blunder[0] - 1
            sum_remaining_pieces += gameplay[move][player_id][3]

        return sum_remaining_pieces / blunders_total
