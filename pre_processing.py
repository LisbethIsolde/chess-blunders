import re
import pandas as pd


class PreProcessing:
    """
    Pre processing module to convert gameplay data into a list and extract blunders
    """

    @staticmethod
    def convert_gameplay(gameplay_str):
        """
        Function that splits the string of the Gameplay into a list with the structure:
        [[move_white, engine_evaluation, interpretation],[move_black, engine_evaluation, interpretation]] x game_length

        Parameters
        ----------
        gameplay_str : str
            string of a chess game

        Return
        ------
        gameplay_list : list
            list of moves in a chess match in the format per list entry:
            [[move_white, engine_evaluation, interpretation],[move_black, engine_evaluation, interpretation]]
        """

        gameplay_list = re.split('\d+\.\s', gameplay_str)[1:]  # separate the moves by "[number].", eg "1.", "22.", etc.
        for i, full_move in enumerate(gameplay_list):
            # split into white/black moves by the indicator for the second half-move "[number]...", eg. "1..."
            gameplay_list[i] = re.split('\s\d+\.\.\.\s', full_move)  
            for j, half_move in enumerate(gameplay_list[i]):
                # after the actual move (e.g. Qxe5) there a " " before the evaluation
                gameplay_list[i][j] = [half_move.split(" ")[0]]
                if re.findall('#\-?\d+|\-?\d+\.\d+', half_move):  # catch last moves before mate
                    gameplay_list[i][j].append(re.findall('#\-?\d+|\-?\d+\.\d+', half_move)[0])
                else:
                    gameplay_list[i][j].append("")
                ann = list(filter(None, re.findall('[!|?]*', gameplay_list[i][j][0])))
                if ann:  # find annotation symbols
                    gameplay_list[i][j].append(ann[0])
                else:
                    gameplay_list[i][j].append("")
                gameplay_list[i][j][0] = "".join(re.findall('\w|\-', gameplay_list[i][j][0]))  # remove annotation symbols from move

        return gameplay_list

    @staticmethod
    def preprocess_gameplay(df):
        """
        Calls helper methods to convert the gameplay column into a list

        Parameters
        ----------
        df : df
            pandas dataframe
        """

        df.Gameplay = df.Gameplay.apply(PreProcessing.convert_gameplay)
        df.Gameplay = df.Gameplay.apply(PreProcessing.add_remaining_pieces)
        df["Blunders"] = df.Gameplay.apply(PreProcessing.find_blunder)

    @staticmethod
    def convert_df(df):
        """
        Deletes unnecessary data and converts strings into timestamps inplace.

        Parameters
        ----------
        df : df
            pandas dataframe
        """

        # convert time
        df.insert(7, "UTCDateTime", df["UTCDate"] + "-" + df["UTCTime"])
        df.UTCDateTime = pd.to_datetime(df.UTCDateTime, format="%Y.%m.%d-%H:%M:%S")
        df.drop(["Site", "Date", "Round", "UTCDate", "UTCTime"], axis=1, inplace=True)

        # delete anonymous games 
        df.drop(df.loc[df["WhiteElo"] == "?"].index, inplace=True)
        df.drop(df.loc[df["BlackElo"] == "?"].index, inplace=True)

        # set datatype of ELO to integer
        df.WhiteElo = df.WhiteElo.astype(int)
        df.BlackElo = df.BlackElo.astype(int)

        # replace result with more meaningful values
        df.Result.replace(to_replace="1-0", value="w", inplace=True)
        df.Result.replace(to_replace="0-1", value="b", inplace=True)
        df.Result.replace(to_replace="1/2-1/2", value="d", inplace=True)

    @staticmethod
    def find_blunder(gameplay_list):
        """
        Function parses a list of moves from a match into a list of blunders:
        [move_number, player, move_string, annotation, diff_in_eval] x number of blunders

        Parameters
        ----------
        gameplay_list : list
            list of moves of a chess match

        Return
        ------
        blunder_list : list
            list of blunders of a chess match
        """

        blunders = []
        eval_b_before = 0
        for i, move in enumerate(gameplay_list):
            if len(move) < 2:
                break
            [move_w, eval_w, ann_w, _], [move_b, eval_b, ann_b, _] = move

            if "-#" in eval_w:
                eval_w = float("-inf")
            elif "#" in eval_w:
                eval_w = float("inf")
            elif eval_w:
                eval_w = float(eval_w)

            if "-#" in eval_b:
                eval_b = float("inf")
            elif "#" in eval_b:
                eval_b = float("-inf")
            elif eval_b:
                eval_b = float(eval_b)
            else:
                eval_b = eval_w

            if ann_w == "?" or ann_w == "??" or ann_w == "?!":
                blunders.append([i+1, "w", move_w, ann_w, eval_b_before - eval_w])
            if ann_b == "?" or ann_b == "??" or ann_b == "?!":
                blunders.append([i + 1, "b", move_b, ann_b, - (eval_w - eval_b)])

            eval_b_before = eval_b

        return blunders

    @staticmethod
    def add_remaining_pieces(gameplay_list):
        """
        Function that takes a list with moves in a game, and transforms it
        into a list of moves, but includes the number of pieces a player has left

        Parameters
        ----------
        gameplay_list : list
            list of moves in a chess match
        Return
        ------
        modified_gameplay_list : list
            list of moves in a chess match, including the remaining pieces of the players
        """

        # both players start with 16 pieces on the board
        num_white_pieces = 16
        num_black_pieces = 16

        # we want to add number of remaining pieces to all moves
        modified_gameplay_list = gameplay_list

        for i, move in enumerate(gameplay_list):
            # moves, where both black and white did a move
            if len(move) == 2:
                white_move, black_move = move
                # if there is an "x" in the move string, this means a piece was captured,
                # so we reduce the current number of pieces of the player

                if "x" in white_move[0]:
                    num_black_pieces -= 1
                if "x" in black_move[0]:
                    num_white_pieces -= 1

                modified_gameplay_list[i][1].append(num_black_pieces)
            else: # move, where only white did a move (mostly the last move of game)
                if "x" in move[0][0]:  
                    num_black_pieces -= 1

            if len(gameplay_list) >= i + 2:
                modified_gameplay_list[i+1][0].append(num_white_pieces)

        modified_gameplay_list[0][0].append(16)  # add initial value for white

        return modified_gameplay_list



