import pandas as pd
import numpy as np
from time import strftime
from math import floor
import time


from nba_api.stats.endpoints import playergamelogs


class nbaFantasyModel():

    def __init__(self):
        #check if current year's season has started
        current_year = int(strftime("%Y"))
        if int(strftime("%m")) < 11:
            current_year -= 1

        #initialize dataframe of all game data with current season
        season = str(current_year) + "-" + str(current_year + 1)[2:]
        player_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season)
        pgl_df = player_game_logs.get_data_frames()[0]

        #add past 9 seasons to dataframe, use last 10 seasons
        for n in range(9):
            current_year -= 1
            season = str(current_year) + "-" + str(current_year + 1)[2:]
            player_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season)
            pgl = player_game_logs.get_data_frames()[0]
            pgl_df = pd.concat([pgl_df, pgl])

        #drop unneccessary dataframe columns
        pgl_df = pgl_df.drop(["SEASON_YEAR", "PLAYER_ID", "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "WL", "BLKA", "PF", "PFD", "DD2", "TD3", "GP_RANK", "W_RANK", "L_RANK", "W_PCT_RANK",
                        "MIN_RANK", "FGM_RANK", "FGA_RANK", "FG_PCT_RANK", "FG3M_RANK", "FG3A_RANK", "FG3_PCT_RANK", "FTM_RANK", "FTA_RANK", "FT_PCT_RANK", "OREB_RANK", "DREB_RANK",
                        "REB_RANK", "AST_RANK", "TOV_RANK", "STL_RANK", "BLK_RANK", "BLKA_RANK", "PF_RANK", "PFD_RANK", "PTS_RANK", "PLUS_MINUS_RANK", "NBA_FANTASY_PTS_RANK",
                        "DD2_RANK", "TD3_RANK", "NICKNAME", "PLUS_MINUS", "WNBA_FANTASY_PTS", "WNBA_FANTASY_PTS_RANK", "AVAILABLE_FLAG"], axis=1)

        #split dataframe into X and Y
        self.Y = np.reshape(pgl_df['NBA_FANTASY_PTS'].to_numpy(copy=True), (-1, 1))
        self.X = pgl_df.loc[:, pgl_df.columns != 'NBA_FANTASY_PTS']

        print(self.X.columns)

        #one hot encode categorical data
        one_hot_encoded = pd.get_dummies(self.X, columns=['PLAYER_NAME', 'MATCHUP'])
        print(one_hot_encoded)

        #augment bias column to X
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))

        #random split into training and validation sets
        indices = np.random.permutation(range(len(self.Y)))
        split_ind = floor(0.9 * len(indices))
        train_indices, val_indices = indices[0:split_ind], indices[split_ind:]
        self.train_X, self.val_X = self.X[train_indices], self.X[val_indices]
        self.train_Y, self.val_Y = self.Y[train_indices], self.Y[val_indices]

        #initialize size parameters
        self.n = self.train_X.shape[0]
        self.k = self.train_X.shape[1]

        #initialize weights vector
        self.weights = np.zeros((self.k, 1))

    def training(self, alpha_range, lam_range, nepoch, epsilon):

        best_mse = np.inf
        best_alpha, best_lam = None, None
        train_losses = []
        val_losses = []

        #grid search on alpha and lambda
        for alpha in np.geomspace(alpha_range[0], alpha_range[1], num=10):
            for lam in np.geomspace(lam_range[0], lam_range[1], num=10):

                #initialize random weights
                weights = np.random.randn(self.k, 1) / 100
                for epoch in range(nepoch):
                    #random permutation of indices to create sample set
                    indices = np.random.permutation(self.n)
                    for i in indices:
                        xi = self.train_Xn[i:i+1]
                        yi = self.train_Y[i:i+1]
                        predict = np.dot(xi, self.weights)
                        weights += alpha * np.clip((np.dot(xi.T, yi - predict) - lam * self.weights), -1e-2, 1e-2)

                    mse_train = np.mean(np.square(np.dot(self.train_X, weights) - self.train_Y))
                    mse_val = np.mean(np.square(np.dot(self.val_X, weights) - self.train_Y))
                    train_losses.append(mse_train)
                    val_losses.append(mse_val)

                    #update hyperparameters for new lowest MSE
                    if mse_train < best_mse:
                        best_mse = mse_train
                        best_alpha = alpha
                        best_lam = lam
                        self.weights = weights.copy()

                    #premature stop if MSE is less than given bound epsilon
                    if mse_train < epsilon:
                        print(f"Premature stop, epoch: {epoch}, MSE: {mse_train}")
                        return best_alpha, best_lam, train_losses, val_losses, best_mse

        return best_alpha, best_lam, train_losses, val_losses, best_mse

    def testing(self, testX):
        #augment bias column to test set
        testX = np.hstack((np.ones((testX.shape[0], 1)), testX))
        testY = np.dot(testX, self.weights)
        return testY

model = nbaFantasyModel()

#Compute training time
#start = time.time()
#alpha, lam, train_losses, val_losses, lowest_mse = model.training([10**-10, 10], [1, 1e10], 300, 1e-5)
#end = time.time()

#testY = model.testing(model.test)
#mse = np.mean(np.square(testY - model.y))

#print(f"Training time: {end-start:.2f} seconds\nBest alpha: {alpha}\nBest lambda: {lam}\nLowest MSE: {lowest_mse}")
#print(f"Testing MSE: {mse}")

#choose which regression model works best
#save model as file
#load model
#graphs
#figure out how to provide input
#use multiple independent variables
#processing categorial variables - one hot encoding
#add weight to more recent games
