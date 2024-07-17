import pandas as pd
import numpy as np
from time import strftime

from nba_api.stats.endpoints import playergamelogs


class nbaFantasyModel():

    #pull data from api and split into training and testing features
    def __init__(self):
        #check if current years season has started
        current_year = int(strftime("%Y"))
        if int(strftime("%m")) < 11:
            current_year -= 1

        #initializes dataframe of all game data with current season
        season = str(current_year) + "-" + str(current_year + 1)[2:]
        player_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season)
        pgl_df = player_game_logs.get_data_frames()[0]

        #adds past 5 seasons to dataframe
        for n in range(5):
            current_year -= 1
            season = str(current_year) + "-" + str(current_year + 1)[2:]
            player_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season)
            pgl = player_game_logs.get_data_frames()[0]
            pgl_df = pd.concat([pgl_df, pgl])

        #split data into training and validation sets





model = nbaFantasyModel()


#train model

#save best model

#predict fantasy value

#regression