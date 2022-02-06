from tqdm import tqdm
import csv
import re


class DataLoader:
    """
    Dataloader module to load pgn files into csv files
    """

    @staticmethod
    def create_csv(path_databases, databases, path_csv, name_csv, stockfish_analysis):
        """
        Reads one or multiple lichess database.pgn and creates a csv file containing
        all games with Stockfish analysis evaluation

        Parameters
        ----------
        path_databases : str
            path to .pgn file
        databases : list
            list that contains names of all png databases
        path_csv : str
            path where csv file gets stored
        name_csv : str
            name of new csv file
        stockfish_analysis : bool
            if true, only games with Stockfish analysis are extracted
        """

        # create new csv file
        if path_csv.endswith("/"):
            path_csv = path_csv[:-1]

        csv_header = ["Event", "Site", "Date", "Round", "White", "Black", "Result", "UTCDate", "UTCTime", "WhiteElo",
                      "BlackElo", "WhiteRatingDiff", "BlackRatingDiff", "WhiteTitle", "BlackTitle", "ECO",
                      "Opening", "TimeControl", "Termination", "Gameplay"]

        with open(path_csv + "/" + name_csv, "wt") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=",")
            filewriter.writerow(csv_header)

        csv_dict = dict(Event=0, Site=1, Date=2, Round=3, White=4, Black=5, Result=6, UTCDate=7, UTCTime=8,
                        WhiteElo=9, BlackElo=10, WhiteRatingDiff=11, BlackRatingDiff=12, WhiteTitle=13,
                        BlackTitle=14, ECO=15, Opening=16, TimeControl=17, Termination=18, Gameplay=19)

        # read all png databases
        if path_databases.endswith("/"):
            path_databases = path_csv[:-1]

        i = 0
        for db in databases:
            f = open(path_databases + "/" + db, "r")
            lines = f.readlines()

            data = [""] * 20
            for line in tqdm(lines):
                if line != "\n":
                    item = line.split()[0][1:]
                    if item == ".":  # Gameplay
                        item = "Gameplay"
                        data[csv_dict[item]] = line
                    else:
                        key = re.findall('"([^"]*)"', line)

                        if key and item != "LichessId":  # some games don't have any Gameplay, and don't save LichessId
                            data[csv_dict[item]] = key[0]
                else:
                    continue

                # add data to csv, since 'Gameplay' is last item of a dataset
                if item == "Gameplay":
                    if stockfish_analysis and "eval" in line:
                        with open(path_csv + "/" + name_csv, "a") as csvfile:
                            filewriter = csv.writer(csvfile, delimiter=",")
                            filewriter.writerow(data)
                            i += 1
                    elif not stockfish_analysis:
                        with open(path_csv + "/" + name_csv, "a") as csvfile:
                            filewriter = csv.writer(csvfile, delimiter=",")
                            filewriter.writerow(data)

                    data = [""] * 20

        print(i, " games found")

